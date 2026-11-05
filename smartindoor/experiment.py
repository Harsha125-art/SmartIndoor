"""Run grouped-CV feature/model comparisons and one final validation pass."""

import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from smartindoor.data import load_indoor_data, load_validation_data
from smartindoor.feature_selection import STRATEGIES, compute_feature_scores, rank_from_scores
from smartindoor.models import MODEL_NAMES, make_classifier, make_pipeline


FEATURE_COUNTS = (25, 50, 100, 150)


def _cross_validation_splits(y, groups, n_splits):
    """
    Produce cross-validation splits.
    
    When USERID metadata is available, StratifiedGroupKFold is preferred because
    it groups measurements by user, ensuring the model generalizes across distinct
    human surveyors. 3 folds is the principled setting for UJIIndoorLoc because
    there are 18 users; with 3 folds (~12 training users per fold), every training fold
    maintains coverage of all 13 building-floor classes without losing any class.
    """
    if groups is not None:
        grouped = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        splits = list(grouped.split(np.zeros(len(y)), y, groups))
        if all(set(y[train]) == set(y) for train, _ in splits):
            return splits, "StratifiedGroupKFold (USERID groups)"

    stratified = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(stratified.split(np.zeros(len(y)), y)), "StratifiedKFold fallback"


def _score_predictions(y_true, y_pred, classes):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        )),
    }


def _run_cross_validation(data, n_splits):
    y = data.y_train.to_numpy()
    groups = data.groups.to_numpy() if data.groups is not None else None
    splits, protocol = _cross_validation_splits(y, groups, n_splits)
    predictions = {
        (strategy, count, model): np.empty(len(y), dtype=object)
        for strategy in STRATEGIES
        for count in FEATURE_COUNTS
        for model in MODEL_NAMES
    }

    print(f"\n--- Starting Cross-Validation ({protocol}, {len(splits)} folds) ---", flush=True)
    for fold_number, (train_indices, test_indices) in enumerate(splits, start=1):
        fold_start = perf_counter()
        print(f"CV Fold {fold_number}/{len(splits)}: Preprocessing and scoring features...", flush=True)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_train = imputer.fit_transform(data.X_train.iloc[train_indices])
        X_test = imputer.transform(data.X_train.iloc[test_indices])
        y_train, y_test = y[train_indices], y[test_indices]

        mi_scores, anova_scores = compute_feature_scores(X_train, y_train)
        rankings = {
            strategy: rank_from_scores(X_train, mi_scores, anova_scores, strategy)
            for strategy in STRATEGIES
        }
        print(f"CV Fold {fold_number}/{len(splits)}: Fitting model configurations...", flush=True)
        for strategy, ranking in rankings.items():
            for count in FEATURE_COUNTS:
                selected = ranking[:count]
                fold_train = X_train[:, selected]
                fold_test = X_test[:, selected]
                for model_name in MODEL_NAMES:
                    if model_name == "Random Forest":
                        model_train, model_test = fold_train, fold_test
                    else:
                        scaler = StandardScaler()
                        model_train = scaler.fit_transform(fold_train)
                        model_test = scaler.transform(fold_test)
                    estimator = make_classifier(model_name)
                    estimator.fit(model_train, y_train)
                    key = (strategy, count, model_name)
                    predictions[key][test_indices] = estimator.predict(model_test)
        print(f"CV Fold {fold_number}/{len(splits)} completed in {perf_counter() - fold_start:.1f}s", flush=True)

    rows = []
    for (strategy, count, model_name), predicted in predictions.items():
        scores = _score_predictions(y, predicted, data.classes)
        rows.append(
            {
                "feature_selection": strategy,
                "feature_limit": count,
                "model": model_name,
                **scores,
                "cv_protocol": protocol,
                "folds": len(splits),
            }
        )
    return pd.DataFrame(rows), protocol


def _save_comparison_plots(results: pd.DataFrame, output_dir: Path):
    model_scores = (
        results.groupby("model", as_index=False)["macro_f1"].max()
        .sort_values("macro_f1", ascending=False)
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(model_scores["model"], model_scores["macro_f1"], color="#2b5c8f", width=0.55)
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axis.set_title("Best Grouped-CV Macro F1 by Model", fontsize=12, fontweight="bold", pad=12)
    axis.set_ylabel("Macro F1 Score", fontsize=11)
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=15, labelsize=10)
    axis.grid(axis="y", alpha=0.3, linestyle="--")
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=200)
    plt.close(figure)

    selection_scores = (
        results.groupby(["feature_selection", "feature_limit"], as_index=False)["macro_f1"]
        .mean()
    )
    figure, axis = plt.subplots(figsize=(9, 5.5))
    markers = {
        "mutual_information": "o",
        "anova": "s",
        "hybrid": "^",
        "correlation_filtered_hybrid": "D",
    }
    colors = {
        "mutual_information": "#1f77b4",
        "anova": "#ff7f0e",
        "hybrid": "#2ca02c",
        "correlation_filtered_hybrid": "#d62728",
    }
    for strategy, group in selection_scores.groupby("feature_selection"):
        group = group.sort_values("feature_limit")
        label = strategy.replace("_", " ").title()
        marker = markers.get(strategy, "o")
        color = colors.get(strategy, None)
        axis.plot(
            group["feature_limit"],
            group["macro_f1"],
            marker=marker,
            markersize=7,
            linewidth=2,
            color=color,
            label=label,
        )
    axis.set_title(
        "Feature Selection Strategy Comparison (Mean Across Models)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axis.set_xlabel("Maximum Selected Wi-Fi Features", fontsize=11)
    axis.set_ylabel("Grouped-CV Macro F1 Score", fontsize=11)
    axis.set_ylim(0, 1.0)
    axis.set_xticks(FEATURE_COUNTS)
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    figure.tight_layout()
    figure.savefig(output_dir / "feature_selection_comparison.png", dpi=200)
    plt.close(figure)


def run_experiment(
    data_dir: Path | str = "data",
    output_dir: Path | str = "output/smartindoor",
    n_splits: int = 3,
):
    """Run all configured comparisons and save the selected calibrated model."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_indoor_data(data_dir)
    print(
        f"Loaded {len(data.y_train):,} training rows, "
        f"{len(data.wap_columns)} WAP features, {len(data.classes)} classes.",
        flush=True,
    )

    results, protocol = _run_cross_validation(data, n_splits)
    results.to_csv(output_dir / "cross_validation_results.csv", index=False)
    _save_comparison_plots(results, output_dir)
    best = results.sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).iloc[0]
    strategy = str(best["feature_selection"])
    feature_limit = int(best["feature_limit"])
    model_name = str(best["model"])
    print(
        "\n=========================================\n"
        f"Best grouped-CV configuration:\n"
        f"  Model: {model_name}\n"
        f"  Feature Selection: {strategy}\n"
        f"  Selected Features: {feature_limit}\n"
        f"  Grouped-CV Macro F1: {best['macro_f1']:.4f}\n"
        f"  Grouped-CV Accuracy: {best['accuracy']:.4f}\n"
        "=========================================\n",
        flush=True,
    )

    # The official validation CSV is first read here, after CV-based selection.
    print("Loading official untouched validation dataset...", flush=True)
    data.X_validation, data.y_validation = load_validation_data(
        data_dir, data.wap_columns
    )
    if not set(data.classes).issubset(set(data.y_validation.unique())):
        raise ValueError("The official validation split does not contain every training class.")
    print(f"Loaded {len(data.y_validation):,} official validation rows for final evaluation.", flush=True)

    print("Fitting final pipeline with sigmoid probability calibration on training set...", flush=True)
    final_pipeline = make_pipeline(model_name, strategy, feature_limit)
    calibrated_model = CalibratedClassifierCV(
        estimator=final_pipeline, method="sigmoid", cv=3, ensemble=False
    )
    calibrated_model.fit(data.X_train, data.y_train)

    print("Evaluating against official validation set...", flush=True)
    validation_prediction = calibrated_model.predict(data.X_validation)
    validation_scores = _score_predictions(
        data.y_validation, validation_prediction, data.classes
    )
    validation_report = classification_report(
        data.y_validation,
        validation_prediction,
        labels=data.classes,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame.from_dict(
        {label: validation_report[label] for label in data.classes}, orient="index"
    )
    per_class.index.name = "class"
    per_class.to_csv(output_dir / "per_class_metrics.csv")
    matrix = confusion_matrix(
        data.y_validation, validation_prediction, labels=data.classes
    )
    pd.DataFrame(matrix, index=data.classes, columns=data.classes).to_csv(
        output_dir / "confusion_matrix.csv", index_label="true_class"
    )
    pd.DataFrame([validation_scores]).to_csv(
        output_dir / "official_validation_metrics.csv", index=False
    )
    joblib.dump(calibrated_model, output_dir / "best_model.joblib")

    fitted_pipeline = calibrated_model.calibrated_classifiers_[0].estimator
    selected_indices = fitted_pipeline.named_steps["selector"].indices_
    selected_features = [data.wap_columns[index] for index in selected_indices]
    metadata = {
        "project": "SmartIndoor",
        "target": "combined BUILDINGID_FLOOR",
        "classes": data.classes,
        "model": model_name,
        "feature_selection": strategy,
        "feature_limit": feature_limit,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "wap_columns": data.wap_columns,
        "calibration": "sigmoid calibration with 3-fold CV on training data",
        "cross_validation": protocol,
        "training_rows": len(data.y_train),
        "official_validation_rows": len(data.y_validation),
        "official_validation_metrics": validation_scores,
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Annotated Confusion Matrix
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="GnBu")
    axis.set_title("Official Validation Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    axis.set_xlabel("Predicted Class (Building_Floor)", fontsize=11, labelpad=8)
    axis.set_ylabel("True Class (Building_Floor)", fontsize=11, labelpad=8)
    axis.set_xticks(np.arange(len(data.classes)))
    axis.set_yticks(np.arange(len(data.classes)))
    axis.set_xticklabels(data.classes, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(data.classes, fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    # Add numeric annotations to cells
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val > 0:
                axis.text(
                    j,
                    i,
                    f"{val:d}",
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8,
                    fontweight="bold" if i == j else "normal",
                )

    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(figure)

    print("\n=== Official Validation Results ===", flush=True)
    for metric, value in validation_scores.items():
        print(f"  {metric}: {value:.4f}", flush=True)
    print(f"\nSaved results and model to: {output_dir.resolve()}", flush=True)
    print(f"Experiment completed in {perf_counter() - started:.1f} seconds", flush=True)
    return results, validation_scores, metadata
