"""Compare classical architectures with 5-fold person-grouped CV.

Only trainingData.csv is read here. The official test file stays unread, so the
choice of final model cannot depend on it.
"""

import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from smartindoor.data import load_indoor_data
from smartindoor.models import TwoStageClassifier, make_pipeline


N_FOLDS = 5
ALL_FEATURES = 520
SVM = "SVM (RBF)"
BOOSTING = "Gradient Boosting"

# Simplicity order for the selection rule, simplest first.
MODEL_ORDER = (SVM, BOOSTING)
SELECTION_ORDER = (
    "all",
    "anova",
    "mutual_information",
    "hybrid",
    "correlation_filtered_hybrid",
)

# (name, stages, model, feature selection, number of access points)
CONFIGURATIONS = (
    ("Baseline: SVM, all access points", 1, SVM, "all", ALL_FEATURES),
    ("SVM, ANOVA", 1, SVM, "anova", 150),
    ("SVM, Mutual Information", 1, SVM, "mutual_information", 150),
    ("SVM, hybrid", 1, SVM, "hybrid", 150),
    ("SVM, filtered hybrid", 1, SVM, "correlation_filtered_hybrid", 150),
    ("Gradient boosting, ANOVA", 1, BOOSTING, "anova", 150),
    ("Two-stage SVM, ANOVA", 2, SVM, "anova", 150),
    ("Two-stage SVM, filtered hybrid", 2, SVM, "correlation_filtered_hybrid", 150),
    ("Two-stage gradient boosting, ANOVA", 2, BOOSTING, "anova", 150),
)


def make_folds(y, groups, start_seed=0):
    """Return grouped, stratified folds where every training side keeps all classes."""
    placeholder = np.zeros(len(y))
    classes = set(y)
    for seed in range(start_seed, start_seed + 200):
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        folds = list(splitter.split(placeholder, y, groups))
        if all(set(y[train]) == classes for train, _ in folds):
            return folds, seed
    raise ValueError("No grouped folds keep every class in training.")


def _build(stages, model_name, selection, n_features, relative_input):
    # Ranking every access point and keeping all of them applies no selection.
    strategy = "anova" if selection == "all" else selection
    if stages == 2:
        return TwoStageClassifier(model_name, strategy, n_features, relative_input)
    return make_pipeline(model_name, strategy, n_features, relative_input)


def _score(y_true, y_pred):
    # A held-out fold may lack some classes, so macro F1 uses those it contains.
    present = sorted(set(y_true))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "macro_precision": precision(y_true, y_pred, present),
        "macro_recall": recall(y_true, y_pred, present),
        "building_accuracy": float(
            np.mean([true[0] == pred[0] for true, pred in zip(y_true, y_pred)])
        ),
    }


def precision(y_true, y_pred, labels):
    return precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def recall(y_true, y_pred, labels):
    return recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def select_configuration(summary: pd.DataFrame) -> pd.Series:
    """Take the simplest configuration within one SD of the best mean macro F1."""
    leader = summary.sort_values("macro_f1_mean", ascending=False).iloc[0]
    threshold = leader["macro_f1_mean"] - leader["macro_f1_std"]
    eligible = summary[summary["macro_f1_mean"] >= threshold]
    return eligible.sort_values(
        ["stages", "model_rank", "selection_rank", "relative_input"]
    ).iloc[0]


def run_comparison(data_dir="data", output_dir="output/smartindoor"):
    started = perf_counter()
    data = load_indoor_data(data_dir)
    X = data.X_train.to_numpy(dtype=float)
    y = data.y_train.to_numpy().astype(str)
    groups = data.groups.to_numpy().astype(str)
    folds, seed = make_folds(y, groups)
    print(f"{len(y):,} rows, {len(set(groups))} people, {N_FOLDS} folds (seed {seed})", flush=True)
    for number, (train, held_out) in enumerate(folds, start=1):
        if set(groups[train]) & set(groups[held_out]):
            raise ValueError("A person appears on both sides of a fold.")
        print(
            f"  Fold {number}: {len(held_out):,} rows held out, "
            f"{len(set(y[held_out]))} classes, users {sorted(set(groups[held_out]), key=int)}",
            flush=True,
        )

    rows = []
    per_class = []
    details = []
    for name, stages, model_name, selection, n_features in CONFIGURATIONS:
        for relative_input in (False, True):
            label = f"{name}{' + relative input' if relative_input else ''}"
            config_start = perf_counter()
            pooled = np.empty(len(y), dtype=object)
            for number, (train, held_out) in enumerate(folds, start=1):
                model = _build(stages, model_name, selection, n_features, relative_input)
                with warnings.catch_warnings():
                    # ANOVA warns about access points that are constant within a class.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    model.fit(X[train], y[train])
                predicted = np.asarray(model.predict(X[held_out])).astype(str)
                pooled[held_out] = predicted
                rows.append({"configuration": label, "fold": number, **_score(y[held_out], predicted)})
            pooled = pooled.astype(str)
            for cls in sorted(set(y)):
                per_class.append(
                    {
                        "configuration": label,
                        "class": cls,
                        "precision": precision(y, pooled, [cls]),
                        "recall": recall(y, pooled, [cls]),
                        "f1": f1_score(y, pooled, labels=[cls], average="macro", zero_division=0),
                    }
                )
            details.append(
                {
                    "configuration": label,
                    "stages": stages,
                    "model": model_name,
                    "model_rank": MODEL_ORDER.index(model_name),
                    "selection": selection,
                    "selection_rank": SELECTION_ORDER.index(selection),
                    "n_features": n_features,
                    "relative_input": relative_input,
                }
            )
            print(f"Finished {label} in {perf_counter() - config_start:.0f}s", flush=True)

    folds_frame = pd.DataFrame(rows)
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "building_accuracy"]
    summary = folds_frame.groupby("configuration", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.join(pd.DataFrame(details).set_index("configuration"))
    summary = summary.sort_values("macro_f1_mean", ascending=False)
    chosen = select_configuration(summary)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_frame.to_csv(output_dir / "architecture_comparison_folds.csv", index=False)
    summary.to_csv(output_dir / "architecture_comparison.csv")
    pd.DataFrame(per_class).to_csv(output_dir / "architecture_comparison_per_class.csv", index=False)
    (output_dir / "selected_configuration.json").write_text(
        json.dumps(
            {
                "configuration": chosen.name,
                "stages": int(chosen["stages"]),
                "model": chosen["model"],
                "selection": chosen["selection"],
                "n_features": int(chosen["n_features"]),
                "relative_input": bool(chosen["relative_input"]),
                "cv_seed": seed,
                "macro_f1_mean": float(chosen["macro_f1_mean"]),
                "macro_f1_std": float(chosen["macro_f1_std"]),
                "accuracy_mean": float(chosen["accuracy_mean"]),
                "accuracy_std": float(chosen["accuracy_std"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n=== {N_FOLDS}-fold person-grouped CV (mean +/- SD over folds) ===")
    for label, row in summary.iterrows():
        print(
            f"{label:<55} F1 {row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f}   "
            f"Acc {row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f}   "
            f"Bldg {row['building_accuracy_mean']:.3f}"
        )
    print(f"\nLeader: {summary.index[0]}")
    print(f"Selected by rule: {chosen.name}")
    print(f"Completed in {perf_counter() - started:.0f}s", flush=True)
    return summary, chosen


if __name__ == "__main__":
    run_comparison()
