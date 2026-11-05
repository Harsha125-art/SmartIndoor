"""Fold-fitted Wi-Fi feature ranking and correlation filtering."""

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif


STRATEGIES = (
    "mutual_information",
    "anova",
    "hybrid",
    "correlation_filtered_hybrid",
)


def _clean_scores(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def compute_feature_scores(X: np.ndarray, y: np.ndarray):
    active = np.ptp(X, axis=0) > 0
    mi = np.zeros(X.shape[1], dtype=float)
    f_scores = np.zeros(X.shape[1], dtype=float)
    if active.any():
        mi[active] = _clean_scores(
            mutual_info_classif(X[:, active], y, random_state=42, n_jobs=-1)
        )
        f_scores[active] = _clean_scores(f_classif(X[:, active], y)[0])
    return mi, f_scores


def rank_features(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    correlation_threshold: float = 0.95,
) -> np.ndarray:
    """Return feature indices in descending relevance order, using only X/y."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown feature-selection strategy: {strategy}")
    mi, f_scores = compute_feature_scores(X, y)
    return rank_from_scores(X, mi, f_scores, strategy, correlation_threshold)


def rank_from_scores(
    X: np.ndarray,
    mi: np.ndarray,
    f_scores: np.ndarray,
    strategy: str,
    correlation_threshold: float = 0.95,
) -> np.ndarray:
    """Build a hybrid or correlation-filtered ranking from fold-local scores."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown feature-selection strategy: {strategy}")
    mi = _clean_scores(mi)
    f_scores = _clean_scores(f_scores)
    if strategy == "mutual_information":
        return np.argsort(-mi, kind="stable")
    if strategy == "anova":
        return np.argsort(-f_scores, kind="stable")

    combined_rank = rankdata(-mi, method="average") + rankdata(
        -f_scores, method="average"
    )
    hybrid_order = np.argsort(combined_rank, kind="stable")
    if strategy == "hybrid":
        return hybrid_order

    active = np.ptp(X, axis=0) > 0
    correlations = np.eye(X.shape[1])
    if active.any():
        corr = np.corrcoef(X[:, active], rowvar=False)
        correlations[np.ix_(active, active)] = np.nan_to_num(corr, nan=0.0)
    kept: list[int] = []
    for candidate in hybrid_order:
        if all(abs(correlations[candidate, chosen]) < correlation_threshold for chosen in kept):
            kept.append(int(candidate))
    return np.asarray(kept, dtype=int)


class WiFiFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: str = "hybrid", n_features: int = 50):
        self.strategy = strategy
        self.n_features = n_features

    def fit(self, X, y):
        if self.n_features < 1:
            raise ValueError("n_features must be positive.")
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self.n_features_in_ = X_arr.shape[1]
        self.indices_ = rank_features(X_arr, y_arr, self.strategy)[: self.n_features]
        if self.indices_.size == 0:
            raise ValueError("Feature selection returned no Wi-Fi features.")
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.indices_].to_numpy()
        X_arr = np.asarray(X)
        return X_arr[:, self.indices_]

    def get_support(self, indices: bool = False):
        if indices:
            return self.indices_.copy()
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.indices_] = True
        return support
