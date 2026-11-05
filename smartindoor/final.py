"""Fit the frozen final model and score it once on the official test file."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from smartindoor.compare import SVM, make_folds
from smartindoor.data import load_indoor_data, load_validation_data
from smartindoor.models import make_pipeline


# Chosen with person-grouped CV on the training file only.
STRATEGY = "correlation_filtered_hybrid"
N_FEATURES = 200
PARAMS = {"C": 1.0, "gamma": 0.001}


def _save_confusion_plot(matrix, classes, path):
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="GnBu")
    axis.set_title("Official Test File Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    axis.set_xlabel("Predicted Class (Building_Floor)", fontsize=11, labelpad=8)
    axis.set_ylabel("True Class (Building_Floor)", fontsize=11, labelpad=8)
    axis.set_xticks(np.arange(len(classes)))
    axis.set_yticks(np.arange(len(classes)))
    axis.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(classes, fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value > 0:
                axis.text(
                    column, row, f"{value:d}", ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "black",
                    fontweight="bold" if row == column else "normal",
                )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def run_final(data_dir="data", output_dir="output/smartindoor/final"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_indoor_data(data_dir)
    y_train = data.y_train.to_numpy().astype(str)
    groups = data.groups.to_numpy().astype(str)
    folds, _ = make_folds(y_train, groups)
    model = CalibratedClassifierCV(
        estimator=make_pipeline(SVM, STRATEGY, N_FEATURES, params=PARAMS),
        method="sigmoid",
        cv=folds,
        ensemble=False,
    )
    model.fit(data.X_train, y_train)
    joblib.dump(model, output_dir / "final_model.joblib")

    # The official test file is first read here, after the model is fitted and saved.
    X_test, y_test = load_validation_data(data_dir, data.wap_columns)
    y_true = y_test.to_numpy().astype(str)
    probabilities = model.predict_proba(X_test)
    predicted = model.classes_[probabilities.argmax(axis=1)].astype(str)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true

    classes = data.classes
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=classes, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, predicted, labels=classes, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predicted, labels=classes, average="macro", zero_division=0)),
        "building_accuracy": float(np.mean([t[0] == p[0] for t, p in zip(y_true, predicted)])),
        "correct": int(correct.sum()),
        "rows": len(y_true),
        "mean_confidence": float(confidence.mean()),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)

    report = classification_report(
        y_true, predicted, labels=classes, output_dict=True, zero_division=0
    )
    per_class = pd.DataFrame.from_dict({label: report[label] for label in classes}, orient="index")
    per_class.index.name = "class"
    per_class.to_csv(output_dir / "per_class_metrics.csv")

    matrix = confusion_matrix(y_true, predicted, labels=classes)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        output_dir / "confusion_matrix.csv", index_label="true_class"
    )
    _save_confusion_plot(matrix, classes, output_dir / "confusion_matrix.png")

    selector = model.calibrated_classifiers_[0].estimator.named_steps["selector"]
    (output_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model": SVM,
                "feature_selection": STRATEGY,
                "selected_feature_count": len(selector.indices_),
                "selected_features": [data.wap_columns[i] for i in selector.indices_],
                "params": PARAMS,
                "training_rows": len(y_train),
                "test_rows": len(y_true),
                "calibration": "sigmoid, 5 person-grouped folds on the training file",
                "test_metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Trained on {len(y_train):,} rows, tested on {len(y_true):,} rows")
    print(f"Selected access points: {len(selector.indices_)}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    print("\nPer-class results:")
    print(per_class.round(3).to_string())
    print("\nConfusion matrix:")
    print(pd.DataFrame(matrix, index=classes, columns=classes).to_string())
    for low, high in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        band = (confidence >= low) & (confidence < high)
        if band.any():
            print(f"Confidence {low}-{high}: {band.sum()} rows, accuracy {correct[band].mean():.3f}")


if __name__ == "__main__":
    run_final()
