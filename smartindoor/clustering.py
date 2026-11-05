"""Check whether unsupervised clusters match the building-floor classes.

Only the training file is read. Labels are used to score the clusters, never to form
them. Run with: python -m smartindoor.clustering
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

SEEDS = range(5)
SILHOUETTE_SAMPLE = 4000
SILHOUETTE_K = (2, 3, 4, 5, 8, 10, 13, 16, 20, 25, 30)
FLOORS_PER_BUILDING = {0: 4, 1: 4, 2: 5}


def _kmeans(features, k, seed, n_init=5):
    return KMeans(k, n_init=n_init, random_state=seed).fit_predict(features)


def _score(name, features, k, target, frame):
    """Score k-means against a target, plus floors, users and phones, over several seeds."""
    labels = [_kmeans(features, k, seed) for seed in SEEDS]
    ari = [adjusted_rand_score(target, label) for label in labels]

    def nmi(column):
        return float(np.mean([normalized_mutual_info_score(column, label) for label in labels]))

    return {
        "check": name,
        "clusters": k,
        "ari_mean": float(np.mean(ari)),
        "ari_std": float(np.std(ari)),
        "nmi_target": nmi(target),
        "nmi_floor": nmi(frame["FLOOR"]),
        "nmi_user": nmi(frame["USERID"]),
        "nmi_phone": nmi(frame["PHONEID"]),
    }


def run_clustering(data_dir="data", output_dir="output/smartindoor/final"):
    frame = pd.read_csv(Path(data_dir) / "trainingData.csv")
    waps = [column for column in frame.columns if column.startswith("WAP")]
    signals = frame[waps].replace(100, -100).to_numpy(float)
    signals = signals[:, signals.std(axis=0) > 0]
    labels = (frame["BUILDINGID"].astype(str) + "_" + frame["FLOOR"].astype(str)).to_numpy()
    buildings = frame["BUILDINGID"].to_numpy()

    variance = np.cumsum(PCA(random_state=0).fit(signals).explained_variance_ratio_)
    components = int(np.searchsorted(variance, 0.9) + 1)
    reduced = PCA(n_components=components, random_state=0).fit_transform(signals)
    detected = (signals > -100).astype(float)
    print(f"{len(signals):,} rows, {signals.shape[1]} active access points, {components} PCA components")

    rows = []
    for name, features in (("signal strength", reduced), ("detected or not", detected)):
        rows.append(_score(f"Buildings, {name}", features, 3, buildings, frame))
        rows.append(_score(f"13 classes, {name}", features, 13, labels, frame))
    for building, floors in FLOORS_PER_BUILDING.items():
        mask = buildings == building
        subset = PCA(n_components=50, random_state=0).fit_transform(signals[mask])
        part = frame[mask]
        rows.append(_score(f"Floors in Building {building}, signal strength", subset, floors, part["FLOOR"], part))
    results = pd.DataFrame(rows)

    sample = np.random.default_rng(0).choice(len(reduced), SILHOUETTE_SAMPLE, replace=False)
    silhouette = pd.DataFrame(
        [
            {"clusters": k, "silhouette": silhouette_score(reduced[sample], _kmeans(reduced, k, 0)[sample])}
            for k in SILHOUETTE_K
        ]
    )

    assigned = _kmeans(reduced, 13, 0, n_init=10)
    crosstab = pd.crosstab(pd.Series(labels, name="class"), pd.Series(assigned, name="cluster"))
    purity = crosstab.max(axis=0).sum() / crosstab.values.sum()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "clustering_results.csv", index=False)
    silhouette.to_csv(output_dir / "clustering_silhouette.csv", index=False)
    crosstab.to_csv(output_dir / "clustering_class_by_cluster.csv")

    print(results.round(3).to_string(index=False))
    print(silhouette.round(3).to_string(index=False))
    print(f"Purity with 13 clusters: {purity:.3f}")


if __name__ == "__main__":
    run_clustering()
