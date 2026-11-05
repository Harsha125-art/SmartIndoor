"""UJIIndoorLoc loading and target construction."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class IndoorData:
    X_train: pd.DataFrame
    y_train: pd.Series
    groups: pd.Series
    classes: list[str]
    wap_columns: list[str]


def _read_split(path: Path, wap_columns: list[str] | None = None):
    frame = pd.read_csv(path)
    current_waps = [column for column in frame.columns if column.startswith("WAP")]
    if not current_waps:
        raise ValueError(f"No WAP access-point columns found in {path}.")
    if wap_columns is not None:
        missing = sorted(set(wap_columns) - set(current_waps))
        if missing:
            raise ValueError(f"{path} is missing {len(missing)} training WAP columns.")
        current_waps = wap_columns

    required = {"BUILDINGID", "FLOOR"}
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"{path} is missing target columns: {sorted(absent)}")

    features = frame[current_waps].apply(pd.to_numeric, errors="coerce")
    # UJIIndoorLoc uses 100 to mean that an access point was not detected.
    features = features.mask(features.eq(100), -100.0)
    target = frame["BUILDINGID"].astype(int).astype(str).str.cat(
        frame["FLOOR"].astype(int).astype(str), sep="_"
    )
    return frame, features, target


def load_indoor_data(data_dir: Path | str = "data") -> IndoorData:
    """Load only the training split for cross-validation and model selection."""
    data_dir = Path(data_dir)
    train_frame, X_train, y_train = _read_split(data_dir / "trainingData.csv")
    groups = train_frame["USERID"].astype(str) if "USERID" in train_frame else None
    classes = sorted(y_train.unique(), key=lambda label: tuple(map(int, label.split("_"))))
    if len(classes) != 13:
        raise ValueError(f"Expected 13 building-floor classes, found {len(classes)}.")
    return IndoorData(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        classes=classes,
        wap_columns=list(X_train.columns),
    )


def load_validation_data(data_dir: Path | str, wap_columns: list[str]):
    """Load the official validation split only after CV model selection."""
    _, features, target = _read_split(Path(data_dir) / "validationData.csv", wap_columns)
    return features, target