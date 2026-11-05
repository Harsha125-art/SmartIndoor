"""Tune the finalist models with person-grouped CV on the training file only.

Settings are chosen on one set of folds, then tuned and default versions are compared
on other groupings of people, because the first set's best score is optimistic. The
same 18 people appear in every grouping, so this reduces that optimism but does not
remove it.
"""

import itertools
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed

from smartindoor.compare import BOOSTING, N_FOLDS, SVM, _score, make_folds
from smartindoor.data import load_indoor_data
from smartindoor.feature_selection import rank_features
from smartindoor.models import TwoStageClassifier, make_classifier


SVM_FINALISTS = {
    "SVM, filtered hybrid": "correlation_filtered_hybrid",
    "SVM, ANOVA": "anova",
}
SVM_DEFAULT = {"C": 10.0, "gamma": "scale", "n_features": 150}
SVM_GRID = {
    "C": (1.0, 10.0, 100.0),
    "gamma": ("scale", 0.001, 0.003, 0.01),
    "n_features": (100, 150, 200, 300),
}

BOOSTING_FINALIST = "Two-stage gradient boosting, ANOVA + relative input"
BOOSTING_DEFAULT = {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "n_features": 150}
BOOSTING_RATES = ((0.1, 100), (0.05, 200), (0.1, 200))
BOOSTING_LEAVES = (15, 31)
BOOSTING_FEATURES = (100, 150, 300)
N_CONFIRMATION_PARTITIONS = 3


def _grid(grid):
    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(*grid.values())]


def _boosting_grid():
    return [
        {"learning_rate": rate, "max_iter": rounds, "max_leaf_nodes": leaves, "n_features": count}
        for (rate, rounds), leaves, count in itertools.product(
            BOOSTING_RATES, BOOSTING_LEAVES, BOOSTING_FEATURES
        )
    ]


def _svm_fold(X_train, y_train, X_held, y_held, ranking, settings):
    selected = ranking[: settings["n_features"]]
    scaler = StandardScaler()
    train = scaler.fit_transform(X_train[:, selected])
    held = scaler.transform(X_held[:, selected])
    params = {key: settings[key] for key in ("C", "gamma")}
    model = make_classifier(SVM, params).fit(train, y_train)
    return _score(y_held, model.predict(held).astype(str))


def _evaluate_svm(X, y, folds, rankings, settings_list, label, fold_set):
    """Score every SVM setting on every fold, reusing each fold's feature ranking."""
    tasks = [
        (index, number, settings)
        for index, settings in enumerate(settings_list)
        for number in range(len(folds))
    ]
    scores = Parallel(n_jobs=-1)(
        delayed(_svm_fold)(
            X[folds[number][0]], y[folds[number][0]],
            X[folds[number][1]], y[folds[number][1]],
            rankings[number], settings,
        )
        for _, number, settings in tasks
    )
    return [
        {
            "finalist": label,
            "fold_set": fold_set,
            "setting": index,
            "fold": number + 1,
            **{key: str(value) for key, value in settings.items()},
            **score,
        }
        for (index, number, settings), score in zip(tasks, scores)
    ]


def _evaluate_boosting(X, y, folds, settings_list, fold_set):
    rows = []
    for index, settings in enumerate(settings_list):
        started = perf_counter()
        params = {key: settings[key] for key in ("learning_rate", "max_iter", "max_leaf_nodes")}
        for number, (train, held) in enumerate(folds):
            model = TwoStageClassifier(BOOSTING, "anova", settings["n_features"], True, params)
            model.fit(X[train], y[train])
            rows.append(
                {
                    "finalist": BOOSTING_FINALIST,
                    "fold_set": fold_set,
                    "setting": index,
                    "fold": number + 1,
                    **{key: str(value) for key, value in settings.items()},
                    **_score(y[held], np.asarray(model.predict(X[held])).astype(str)),
                }
            )
        print(
            f"Boosting setting {index + 1}/{len(settings_list)} on {fold_set} folds: "
            f"{perf_counter() - started:.0f}s",
            flush=True,
        )
    return rows


def _rankings(X, y, folds, strategy):
    return [rank_features(X[train], y[train], strategy) for train, _ in folds]


def _best_setting(rows):
    """Return the setting index with the highest mean macro F1."""
    frame = pd.DataFrame(rows)
    return int(frame.groupby("setting")["macro_f1"].mean().idxmax())


def _partition_key(folds, groups):
    return frozenset(frozenset(groups[held]) for _, held in folds)


def _other_partitions(y, groups, tuning_folds, count=N_CONFIRMATION_PARTITIONS):
    """Return fold sets that group people differently from the tuning folds.

    StratifiedGroupKFold can give the same grouping for every seed, so these are built
    directly: people are shuffled, then each joins the fold with the fewest rows. Every
    training side must keep all classes. These folds are balanced by size, not stratified.
    """
    seen = {_partition_key(tuning_folds, groups)}
    people, sizes = np.unique(groups, return_counts=True)
    classes = set(y)
    positions = np.arange(len(y))
    partitions = []
    for seed in range(5000):
        if len(partitions) == count:
            break
        order = np.random.default_rng(seed).permutation(len(people))
        fold_rows = np.zeros(N_FOLDS)
        fold_people = [[] for _ in range(N_FOLDS)]
        for person in order:
            target = int(np.argmin(fold_rows))
            fold_rows[target] += sizes[person]
            fold_people[target].append(people[person])
        folds = []
        for held_people in fold_people:
            held_mask = np.isin(groups, held_people)
            folds.append((positions[~held_mask], positions[held_mask]))
        if not all(set(y[train]) == classes for train, _ in folds):
            continue
        key = _partition_key(folds, groups)
        if key not in seen:
            seen.add(key)
            partitions.append((seed, folds))
    if len(partitions) < count:
        raise ValueError(f"Found only {len(partitions)} different groupings of people.")
    return partitions


def _load_arrays(data_dir):
    data = load_indoor_data(data_dir)
    X = data.X_train.to_numpy(dtype=float)
    y = data.y_train.to_numpy().astype(str)
    groups = data.groups.to_numpy().astype(str)
    return X, y, groups


def _chosen_from_results(results: pd.DataFrame):
    """Read each finalist's best setting back from the saved grid results."""
    chosen = {}
    for label, group in results.groupby("finalist", sort=False):
        best = group.groupby("setting")["macro_f1"].mean().idxmax()
        row = group[group["setting"] == best].iloc[0]
        if label in SVM_FINALISTS:
            gamma = row["gamma"]
            chosen[label] = {
                "C": float(row["C"]),
                "gamma": gamma if gamma == "scale" else float(gamma),
                "n_features": int(row["n_features"]),
            }
        else:
            chosen[label] = {
                "learning_rate": float(row["learning_rate"]),
                "max_iter": int(float(row["max_iter"])),
                "max_leaf_nodes": int(float(row["max_leaf_nodes"])),
                "n_features": int(row["n_features"]),
            }
    return chosen


def run_confirmation(data_dir="data", output_dir="output/smartindoor"):
    """Compare default and tuned settings on groupings of people not used for tuning."""
    started = perf_counter()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    output_dir = Path(output_dir)
    X, y, groups = _load_arrays(data_dir)
    tuning_folds, _ = make_folds(y, groups)
    chosen = _chosen_from_results(pd.read_csv(output_dir / "tuning_results.csv"))
    partitions = _other_partitions(y, groups, tuning_folds)

    rows = []
    for seed, folds in partitions:
        print(f"\nGrouping from seed {seed}:", flush=True)
        for number, (_, held) in enumerate(folds, start=1):
            print(f"  fold {number}: users {sorted(set(groups[held]), key=int)}", flush=True)
        for label, strategy in SVM_FINALISTS.items():
            rankings = _rankings(X, y, folds, strategy)
            for version, settings in (("default", SVM_DEFAULT), ("tuned", chosen[label])):
                for row in _evaluate_svm(X, y, folds, rankings, [settings], label, seed):
                    rows.append({**row, "version": version})
        for version, settings in (("default", BOOSTING_DEFAULT), ("tuned", chosen[BOOSTING_FINALIST])):
            for row in _evaluate_boosting(X, y, folds, [settings], seed):
                rows.append({**row, "version": version})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "tuning_confirmation.csv", index=False)

    print("\n=== Chosen settings ===")
    for label, settings in chosen.items():
        print(f"{label}: {settings}")
    print(f"\n=== {len(partitions)} other groupings of people: default vs tuned ===")
    for label, group in frame.groupby("finalist", sort=False):
        table = group.groupby("version", sort=False)[["macro_f1", "accuracy", "building_accuracy"]].agg(
            ["mean", "std"]
        )
        print(label)
        print(table.round(3).to_string())
        pivot = group.pivot(index=["fold_set", "fold"], columns="version", values="macro_f1")
        diff = pivot["tuned"] - pivot["default"]
        print(
            f"Tuned minus default: {diff.mean():+.3f} (SD {diff.std():.3f}), "
            f"higher in {(diff > 0).sum()}/{len(diff)} folds\n"
        )
    tuned = frame[frame["version"] == "tuned"].pivot(
        index=["fold_set", "fold"], columns="finalist", values="macro_f1"
    )
    print("=== Tuned finalists, macro F1 per grouping ===")
    print(tuned.groupby(level="fold_set").mean().round(3).to_string())
    print("\nPairwise, tuned (row minus column, mean difference; folds where row is higher):")
    for first, second in itertools.combinations(tuned.columns, 2):
        diff = tuned[first] - tuned[second]
        print(
            f"{first} vs {second}: {diff.mean():+.3f} (SD {diff.std():.3f}), "
            f"higher in {(diff > 0).sum()}/{len(diff)}"
        )
    print(f"\nCompleted in {perf_counter() - started:.0f}s", flush=True)


def run_tuning(data_dir="data", output_dir="output/smartindoor"):
    started = perf_counter()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    X, y, groups = _load_arrays(data_dir)
    tuning_folds, tuning_seed = make_folds(y, groups)
    print(f"Tuning fold seed {tuning_seed}", flush=True)
    for number, (_, held) in enumerate(tuning_folds, start=1):
        print(f"  fold {number}: users {sorted(set(groups[held]), key=int)}", flush=True)

    all_rows = []
    chosen = {}

    for label, strategy in SVM_FINALISTS.items():
        print(f"\nTuning {label}...", flush=True)
        settings_list = _grid(SVM_GRID)
        rankings = _rankings(X, y, tuning_folds, strategy)
        rows = _evaluate_svm(X, y, tuning_folds, rankings, settings_list, label, "tuning")
        all_rows.extend(rows)
        best = settings_list[_best_setting(rows)]
        chosen[label] = best
        print(f"Best: {best}  ({perf_counter() - started:.0f}s elapsed)", flush=True)

    print(f"\nTuning {BOOSTING_FINALIST}...", flush=True)
    settings_list = _boosting_grid()
    rows = _evaluate_boosting(X, y, tuning_folds, settings_list, "tuning")
    for row in rows:
        all_rows.append(row)
    best = settings_list[_best_setting(rows)]
    chosen[BOOSTING_FINALIST] = best
    print(f"Best: {best}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_dir / "tuning_results.csv", index=False)
    print(f"Grid search completed in {perf_counter() - started:.0f}s", flush=True)
    run_confirmation(data_dir, output_dir)


if __name__ == "__main__":
    if "--confirm-only" in sys.argv:
        run_confirmation()
    else:
        run_tuning()
