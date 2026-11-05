"""Classifier definitions and leakage-safe pipeline construction."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from smartindoor.feature_selection import STRATEGIES


MODEL_NAMES = ("KNN", "SVM (RBF)", "Random Forest", "LDA")
UNSCALED_MODELS = ("Random Forest", "Gradient Boosting")
NOT_DETECTED = -100.0


def make_classifier(model_name: str, params: dict | None = None):
    """Build a classifier with its default settings, plus any overrides in params."""
    classifier = _default_classifier(model_name)
    return classifier.set_params(**params) if params else classifier


def _default_classifier(model_name: str):
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    if model_name == "SVM (RBF)":
        return SVC(C=10.0, kernel="rbf", cache_size=1000)
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=160, min_samples_leaf=1, n_jobs=-1, random_state=42
        )
    if model_name == "LDA":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if model_name == "Gradient Boosting":
        # Early stopping is off because its internal holdout would mix people.
        return HistGradientBoostingClassifier(
            max_iter=100, early_stopping=False, random_state=42
        )
    raise ValueError(f"Unknown model: {model_name}")


def relative_to_strongest(X):
    """Express each detected signal relative to the strongest one in its scan.

    Phones differ in overall signal level, so this removes a per-scan offset. It uses
    one row at a time and has nothing to fit, so it cannot leak between splits.
    """
    X = np.asarray(X, dtype=float)
    detected = X > NOT_DETECTED
    if not detected.any():
        return X.copy()
    strongest = np.where(detected, X, -np.inf).max(axis=1, keepdims=True)
    strongest[~np.isfinite(strongest)] = NOT_DETECTED
    return np.where(detected, X - strongest, NOT_DETECTED)


def make_pipeline(
    model_name: str,
    strategy: str,
    n_features: int,
    relative_input: bool = False,
    params: dict | None = None,
) -> Pipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown feature-selection strategy: {strategy}")
    from smartindoor.feature_selection import WiFiFeatureSelector

    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if relative_input:
        steps.append(("relative", FunctionTransformer(relative_to_strongest)))
    steps.append(("selector", WiFiFeatureSelector(strategy, n_features)))
    if model_name not in UNSCALED_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", make_classifier(model_name, params)))
    return Pipeline(steps)


class TwoStageClassifier(ClassifierMixin, BaseEstimator):
    """Predict the building, then the floor with a model fitted for that building.

    Each stage selects its own access points, so a floor model ranks them by how well
    they separate floors within its building rather than across the whole campus.
    """

    def __init__(self, model_name, strategy, n_features, relative_input=False, params=None):
        self.model_name = model_name
        self.strategy = strategy
        self.n_features = n_features
        self.relative_input = relative_input
        self.params = params

    def _pipeline(self):
        return make_pipeline(
            self.model_name,
            self.strategy,
            self.n_features,
            self.relative_input,
            self.params,
        )

    def fit(self, X, y):
        labels = np.asarray(y).astype(str)
        buildings = np.array([label.split("_")[0] for label in labels])
        floors = np.array([label.split("_")[1] for label in labels])
        self.classes_ = np.unique(labels)
        self.building_model_ = self._pipeline().fit(X, buildings)
        self.floor_models_ = {}
        for building in np.unique(buildings):
            rows = buildings == building
            self.floor_models_[building] = self._pipeline().fit(X[rows], floors[rows])
        return self

    def predict(self, X):
        buildings = self.building_model_.predict(X).astype(str)
        predicted = np.empty(len(buildings), dtype=object)
        for building, model in self.floor_models_.items():
            rows = buildings == building
            if rows.any():
                floors = model.predict(X[rows]).astype(str)
                predicted[rows] = [f"{building}_{floor}" for floor in floors]
        return predicted.astype(str)
