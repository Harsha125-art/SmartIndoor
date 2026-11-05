"""Predict coordinates and compare classifiers with person-grouped CV.

Default run: training file only. The coordinate method is chosen on the stratified
5-fold split, and the other groupings of people only check that choice.
With --test: scores the chosen method once on the official test file, alongside the
1-nearest-neighbor method from the UJIIndoorLoc paper (Torres-Sospedra et al., 2014).

The paper's positioning error averages only rows placed in the correct building and
floor, so that figure is reported here as well as the error over all rows.
Coordinates are UTM values in meters, so straight-line distance is valid.
"""

import itertools
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.parallel import Parallel, delayed

from smartindoor.compare import _score, make_folds
from smartindoor.data import load_indoor_data
from smartindoor.feature_selection import rank_features
from smartindoor.final import N_FEATURES, PARAMS, STRATEGY
from smartindoor.tune import _other_partitions


COORDINATES = ["LONGITUDE", "LATITUDE"]
STRUCTURES = ("campus", "per_building", "per_floor")
REGRESSORS = ("KNN k=3", "KNN k=5", "KNN k=9", "Random forest", "SVR")
PAPER_METHOD = "Paper method: 1-nearest neighbor, all access points"


def _regressor(name):
    if name.startswith("KNN"):
        return KNeighborsRegressor(n_neighbors=int(name.split("=")[1]), weights="distance")
    if name == "Random forest":
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    if name == "SVR":
        return MultiOutputRegressor(SVR(C=10.0, gamma=0.001))
    raise ValueError(f"Unknown regressor: {name}")


def _region(labels, structure):
    labels = np.asarray(labels).astype(str)
    if structure == "campus":
        return np.array(["all"] * len(labels))
    if structure == "per_building":
        return np.array([label[0] for label in labels])
    return labels


def _predict_coordinates(name, structure, X_train, coords, labels, X_held, predicted_labels):
    """Fit one regressor per region, then route each row by its predicted location."""
    scaler = StandardScaler()
    targets = scaler.fit_transform(coords)
    train_regions = _region(labels, structure)
    held_regions = _region(predicted_labels, structure)
    output = np.zeros((len(X_held), 2))
    for region in np.unique(held_regions):
        rows = train_regions == region
        model = _regressor(name).fit(X_train[rows], targets[rows])
        output[held_regions == region] = model.predict(X_held[held_regions == region])
    return scaler.inverse_transform(output)


def _errors(true_coords, predicted_coords, y_true, y_pred):
    distance = np.linalg.norm(true_coords - predicted_coords, axis=1)
    correct = np.asarray(y_true) == np.asarray(y_pred)
    return {
        "error_all_mean": float(distance.mean()),
        "error_all_median": float(np.median(distance)),
        "error_all_p95": float(np.percentile(distance, 95)),
        "error_correct_location_mean": float(distance[correct].mean()) if correct.any() else np.nan,
        "located_correctly": float(correct.mean()),
    }


def _select(X, y, ranking):
    selected = ranking[: N_FEATURES]
    scaler = StandardScaler().fit(X[:, selected])
    return selected, scaler


def _evaluate_fold(X, y, coords, train, held, partition, fold):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    y_train, y_held = y[train], y[held]
    identity = {"partition": partition, "fold": fold}
    classification = []
    regression = []

    def add_classifier(name, predicted):
        classification.append({**identity, "configuration": name, **_score(y_held, predicted)})

    # Paper method, for both location classes and coordinates.
    nearest = KNeighborsClassifier(n_neighbors=1, n_jobs=1).fit(X[train], y_train)
    paper_labels = nearest.predict(X[held]).astype(str)
    add_classifier(PAPER_METHOD, paper_labels)
    paper_coords = (
        KNeighborsRegressor(n_neighbors=1, n_jobs=1).fit(X[train], coords[train]).predict(X[held])
    )
    regression.append(
        {**identity, "configuration": PAPER_METHOD, **_errors(coords[held], paper_coords, y_held, paper_labels)}
    )

    # Baseline: SVM with default settings and no feature selection.
    scaler = StandardScaler()
    baseline = SVC(C=10.0).fit(scaler.fit_transform(X[train]), y_train)
    add_classifier("Baseline: SVM, all access points", baseline.predict(scaler.transform(X[held])).astype(str))

    # Mutual Information SVM: the setup behind the 90.19% result.
    ranking = rank_features(X[train], y_train, "mutual_information")[:150]
    scaler = StandardScaler()
    model = SVC(C=10.0).fit(scaler.fit_transform(X[train][:, ranking]), y_train)
    add_classifier("SVM, Mutual Information (original)", model.predict(scaler.transform(X[held][:, ranking])).astype(str))

    # The final classifier: tuned SVM with filtered hybrid selection.
    ranking = rank_features(X[train], y_train, STRATEGY)
    selected, scaler = _select(X[train], y_train, ranking)
    X_train = scaler.transform(X[train][:, selected])
    X_held = scaler.transform(X[held][:, selected])
    classifier = SVC(**PARAMS).fit(X_train, y_train)
    predicted = classifier.predict(X_held).astype(str)
    add_classifier("SVM, filtered hybrid, tuned (final)", predicted)

    for structure, name in itertools.product(STRUCTURES, REGRESSORS):
        estimate = _predict_coordinates(name, structure, X_train, coords[train], y_train, X_held, predicted)
        regression.append(
            {**identity, "configuration": f"{name}, {structure}", **_errors(coords[held], estimate, y_held, predicted)}
        )
    return classification, regression


def _load_training(data_dir):
    data = load_indoor_data(data_dir)
    frame = pd.read_csv(Path(data_dir) / "trainingData.csv", usecols=COORDINATES)
    X = data.X_train.to_numpy(dtype=float)
    y = data.y_train.to_numpy().astype(str)
    groups = data.groups.to_numpy().astype(str)
    return data, X, y, groups, frame[COORDINATES].to_numpy(dtype=float)


def run_cv(data_dir="data", output_dir="output/smartindoor/final"):
    started = perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, X, y, groups, coords = _load_training(data_dir)
    stratified, seed = make_folds(y, groups)
    partitions = [("stratified", stratified)] + [
        (f"grouping_{other_seed}", folds)
        for other_seed, folds in _other_partitions(y, groups, stratified)
    ]
    tasks = [
        (name, number, train, held)
        for name, folds in partitions
        for number, (train, held) in enumerate(folds, start=1)
    ]
    print(f"Running {len(tasks)} folds; stratified fold seed {seed}", flush=True)
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_evaluate_fold)(X, y, coords, train, held, name, number)
        for name, number, train, held in tasks
    )
    classification = pd.DataFrame([row for result in results for row in result[0]])
    regression = pd.DataFrame([row for result in results for row in result[1]])
    classification.to_csv(output_dir / "classifier_check.csv", index=False)
    regression.to_csv(output_dir / "coordinate_cv.csv", index=False)

    def label(frame):
        return np.where(frame["partition"] == "stratified", "stratified", "other_groupings")

    print("\n=== Classifiers: mean +/- SD over folds ===")
    summary = classification.assign(folds_used=label(classification)).groupby(
        ["folds_used", "configuration"]
    )[["macro_f1", "accuracy", "building_accuracy"]].agg(["mean", "std"])
    print(summary.round(3).to_string())

    print("\n=== Coordinates: mean +/- SD over folds, meters ===")
    summary = regression.assign(folds_used=label(regression)).groupby(
        ["folds_used", "configuration"]
    )[["error_all_mean", "error_all_median", "error_correct_location_mean"]].agg(["mean", "std"])
    print(summary.round(2).to_string())

    candidates = regression[
        (regression["partition"] == "stratified") & (regression["configuration"] != PAPER_METHOD)
    ]
    chosen = candidates.groupby("configuration")["error_all_mean"].mean().idxmin()
    name, structure = chosen.split(", ")
    (output_dir / "coordinate_method.json").write_text(
        json.dumps({"configuration": chosen, "regressor": name, "structure": structure}, indent=2),
        encoding="utf-8",
    )
    print(f"\nChosen coordinate method: {chosen}")
    print(f"Completed in {perf_counter() - started:.0f}s", flush=True)


def run_test(data_dir="data", output_dir="output/smartindoor/final"):
    """Score the chosen coordinate method and the paper's method on the test file."""
    warnings.filterwarnings("ignore", category=UserWarning)
    output_dir = Path(output_dir)
    data, X, y, _, coords = _load_training(data_dir)
    method = json.loads((output_dir / "coordinate_method.json").read_text(encoding="utf-8"))
    classifier = joblib.load(output_dir / "final_model.joblib")

    test_frame = pd.read_csv(Path(data_dir) / "validationData.csv")
    X_test_frame = test_frame[data.wap_columns].mask(test_frame[data.wap_columns].eq(100), -100.0)
    X_test = X_test_frame.to_numpy(dtype=float)
    y_test = (
        test_frame["BUILDINGID"].astype(int).astype(str) + "_" + test_frame["FLOOR"].astype(int).astype(str)
    ).to_numpy()
    test_coords = test_frame[COORDINATES].to_numpy(dtype=float)
    predicted = classifier.predict(X_test_frame).astype(str)

    ranking = rank_features(X, y, STRATEGY)
    selected, scaler = _select(X, y, ranking)
    estimate = _predict_coordinates(
        method["regressor"], method["structure"],
        scaler.transform(X[:, selected]), coords, y,
        scaler.transform(X_test[:, selected]), predicted,
    )
    rows = [{"method": f"Final: {method['configuration']}", **_errors(test_coords, estimate, y_test, predicted)}]

    # The paper's method, using each way of encoding "not detected".
    raw_train = pd.read_csv(Path(data_dir) / "trainingData.csv", usecols=data.wap_columns)[data.wap_columns]
    raw_test = test_frame[data.wap_columns]
    for encoding, train_values, test_values in (
        ("not detected = +100 (as in the files)", raw_train.to_numpy(float), raw_test.to_numpy(float)),
        ("not detected = -100", X, X_test),
    ):
        labels = KNeighborsClassifier(n_neighbors=1).fit(train_values, y).predict(test_values).astype(str)
        position = KNeighborsRegressor(n_neighbors=1).fit(train_values, coords).predict(test_values)
        rows.append(
            {
                "method": f"{PAPER_METHOD}, {encoding}",
                **_errors(test_coords, position, y_test, labels),
                "accuracy": accuracy_score(y_test, labels),
                "macro_f1": f1_score(y_test, labels, average="macro"),
                "building_errors": int(sum(t[0] != p[0] for t, p in zip(y_test, labels))),
            }
        )
    rows[0].update(
        accuracy=accuracy_score(y_test, predicted),
        macro_f1=f1_score(y_test, predicted, average="macro"),
        building_errors=int(sum(t[0] != p[0] for t, p in zip(y_test, predicted))),
    )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "coordinate_test_results.csv", index=False)
    print(table.round(4).T.to_string())

    distance = np.linalg.norm(test_coords - estimate, axis=1)
    print("\nFinal method, mean error by true building (m):")
    print(pd.Series(distance).groupby(pd.Series(y_test).str[0]).mean().round(2).to_string())


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_test()
    else:
        run_cv()
