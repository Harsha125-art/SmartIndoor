"""Classifier definitions and leakage-safe pipeline construction."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from smartindoor.feature_selection import STRATEGIES


MODEL_NAMES = ("KNN", "SVM (RBF)", "Random Forest", "LDA")


def make_classifier(model_name: str):
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
    raise ValueError(f"Unknown model: {model_name}")


def make_pipeline(model_name: str, strategy: str, n_features: int) -> Pipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown feature-selection strategy: {strategy}")
    from smartindoor.feature_selection import WiFiFeatureSelector

    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("selector", WiFiFeatureSelector(strategy, n_features)),
    ]
    if model_name != "Random Forest":
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", make_classifier(model_name)))
    return Pipeline(steps)