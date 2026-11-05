"""Streamlit interface for a single supplied Wi-Fi fingerprint."""

import json
from pathlib import Path
import re
import sys

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "output" / "smartindoor"
FINAL_DIR = OUTPUT_DIR / "final"
MODEL_PATH = FINAL_DIR / "final_model.joblib"
METADATA_PATH = FINAL_DIR / "model_metadata.json"

st.set_page_config(page_title="SmartIndoor - Wi-Fi Localization", page_icon="⌖", layout="wide")
st.title("SmartIndoor")
st.caption("Wi-Fi Fingerprint Localization · UJIIndoorLoc Building and Floor Classification")

with st.expander("ℹ️ How supplied Wi-Fi fingerprints are processed"):
    st.markdown(
        """
        - **Input Sentinel Mapping:** Any unobserved access point defaults to `-100.0 dBm` (the UJIIndoorLoc sentinel representing non-detection). Actual NaNs are median-imputed.
        - **Feature Selection:** Only the top-ranked discriminative Wi-Fi access points (identified via supervised training without data leakage) are selected from the 520 candidates.
        - **Standardization & Classification:** Selected RSSI signals are scaled and classified into one of the 13 discrete campus building-floor classes (Buildings 0, 1, 2).
        - **Calibrated Confidence:** Posterior probabilities are calculated via sigmoid probability calibration (Platt scaling) fitted on training cross-validation folds.
        - **Operational Scope:** Classifies supplied RSSI fingerprints; does **not** perform live Wi-Fi radio hardware scanning.
        """
    )

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("No trained SmartIndoor model was found. Run `python main.py` first to generate model artifacts.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
all_waps = metadata.get("wap_columns", [])
if not all_waps:
    all_waps = [f"WAP{index:03d}" for index in range(1, 521)]


def normalize_wap_name(name: str) -> str:
    """Normalize WAP names (e.g. 'WAP1' -> 'WAP001')."""
    match = re.match(r"^WAP(\d+)$", str(name).strip(), re.IGNORECASE)
    if match:
        return f"WAP{int(match.group(1)):03d}"
    return str(name).strip()


left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Fingerprint Input")
    mode = st.radio("Input format", ("JSON", "CSV"), horizontal=True)

    default_json = '{\n  "WAP006": -85,\n  "WAP007": -86,\n  "WAP011": -60,\n  "WAP012": -61,\n  "WAP017": -70\n}'

    with st.form("fingerprint_form"):
        if mode == "JSON":
            raw = st.text_area(
                "Observed access points and RSSI (dBm):",
                value=st.session_state.get("input_json_val", default_json),
                height=160,
                help="Enter a JSON object mapping WAP IDs (e.g., 'WAP011' or 'WAP11') to received signal strength values in dBm.",
            )
            uploaded = None
        else:
            uploaded = st.file_uploader(
                "Upload a CSV file with one fingerprint row:",
                type=("csv",),
                help="CSV should contain WAP columns with RSSI values (or 100 for not detected).",
            )
            raw = ""
        submitted = st.form_submit_button("Locate Fingerprint", type="primary")

    if submitted:
        try:
            payload = {}
            if mode == "JSON":
                if not raw or not raw.strip():
                    raise ValueError("JSON input is empty. Please enter WAP observations.")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as decode_err:
                    raise ValueError(f"Invalid JSON syntax: {decode_err}")
                if not isinstance(parsed, dict):
                    raise ValueError("JSON must be a key-value object mapping WAP names to RSSI numbers.")
                payload = {normalize_wap_name(k): v for k, v in parsed.items()}
            elif uploaded is not None:
                try:
                    df_upload = pd.read_csv(uploaded)
                except Exception as csv_err:
                    raise ValueError(f"Unable to read CSV file: {csv_err}")
                if df_upload.empty:
                    raise ValueError("Uploaded CSV is empty.")
                df_upload.columns = [normalize_wap_name(col) for col in df_upload.columns]
                row = df_upload.iloc[0]
                payload = {col: row[col] for col in all_waps if col in row.index}
                if not payload:
                    raise ValueError("No recognizable WAP columns (WAP001 to WAP520) found in the CSV.")
            else:
                raise ValueError("Please select or upload a CSV file containing at least one row.")

            # Validate WAP column names
            unknown = sorted(set(payload.keys()) - set(all_waps))
            if unknown:
                raise ValueError(f"Unknown access-point columns: {', '.join(unknown[:5])}")

            # Construct full 520-dimensional input vector
            vector = pd.DataFrame([[-100.0] * len(all_waps)], columns=all_waps)
            active_count = 0
            for name, value in payload.items():
                if pd.isna(value):
                    continue
                try:
                    numeric = float(value)
                except (ValueError, TypeError):
                    raise ValueError(f"RSSI value for {name} must be a number, got '{value}'.")

                if numeric == 100.0:
                    numeric = -100.0

                if numeric > 0 or numeric < -120:
                    raise ValueError(f"{name} RSSI ({numeric} dBm) is outside valid range [-120 dBm, 0 dBm] (or 100).")

                vector.loc[0, name] = numeric
                if numeric != -100.0:
                    active_count += 1

            predicted_label = model.predict(vector)[0]
            building, floor = map(int, str(predicted_label).split("_"))
            probabilities = model.predict_proba(vector)[0]
            confidence = float(probabilities.max())

            st.session_state["prediction"] = {
                "building": building,
                "floor": floor,
                "confidence": confidence,
                "active": active_count,
            }
        except Exception as error:
            st.error(f"⚠️ Input Error: {error}")

with right:
    st.subheader("Location Estimate")
    prediction = st.session_state.get("prediction")
    if prediction:
        col_b, col_f = st.columns(2)
        with col_b:
            st.metric("🏢 Building", f"Building {prediction['building']}")
        with col_f:
            st.metric("🪜 Floor", f"Floor {prediction['floor']}")

        st.metric("🎯 Calibrated Confidence", f"{prediction['confidence']:.1%}")

        st.markdown("---")
        st.markdown(f"**Best Model:** `{metadata.get('model', 'SVM (RBF)')}`")
        st.markdown(
            f"**Feature Selection:** `{metadata.get('feature_selection', 'Mutual Information').replace('_', ' ').title()}`"
        )
        st.markdown(f"**Selected Features Used:** `{metadata.get('selected_feature_count', 150)}`")
        st.markdown(f"**Detected APs in Fingerprint:** `{prediction['active']}`")
    else:
        st.info("Enter or upload a Wi-Fi fingerprint and click **Locate Fingerprint** to compute building and floor.")
        st.markdown("---")
        st.markdown(f"**Active Model:** `{metadata.get('model', 'SVM (RBF)')}`")
        st.markdown(
            f"**Feature Strategy:** `{metadata.get('feature_selection', 'Mutual Information').replace('_', ' ').title()} ({metadata.get('selected_feature_count', 150)} features)`"
        )

    st.caption("🔒 *Static classification mode: predicts location strictly from provided RSSI vector.*")

st.divider()

st.subheader("Experimental Performance & Validation")
metrics_path = FINAL_DIR / "test_metrics.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path).iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Official Test Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Test Macro F1", f"{metrics['macro_f1']:.4f}")
    col3.metric("Macro Precision", f"{metrics['macro_precision']:.4f}")
    col4.metric("Macro Recall", f"{metrics['macro_recall']:.4f}")

coordinates_path = FINAL_DIR / "coordinate_test_results.csv"
if coordinates_path.exists():
    final_errors = pd.read_csv(coordinates_path).iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Coordinate Error", f"{final_errors['error_all_mean']:.2f} m")
    col2.metric("Median Coordinate Error", f"{final_errors['error_all_median']:.2f} m")
    col3.metric(
        "Mean Error, Correct Building and Floor",
        f"{final_errors['error_correct_location_mean']:.2f} m",
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "📈 Feature Selection Curves",
    "🎯 Confusion Matrix & Class Metrics",
    "📋 5-Fold CV Leaderboard",
])

with tab1:
    comparison_path = OUTPUT_DIR / "architecture_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path, index_col="configuration")
        st.markdown("**Mean Macro F1, 5-Fold Person-Grouped CV (default settings)**")
        st.bar_chart(comparison["macro_f1_mean"], horizontal=True)
    else:
        st.info("Run `python main.py` to generate the model comparison.")

with tab2:
    fs_path = OUTPUT_DIR / "feature_selection_comparison.png"
    if fs_path.exists():
        st.image(str(fs_path), caption="Original 3-Fold Experiment: Feature Selection Strategy vs Number of Access Points")
    else:
        st.info("Run `python main.py` to generate the feature selection comparison chart.")

with tab3:
    cm_path = FINAL_DIR / "confusion_matrix.png"
    per_class_path = FINAL_DIR / "per_class_metrics.csv"
    col_cm, col_pc = st.columns([1, 1], gap="medium")
    with col_cm:
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix on the Official Test File (13 Building-Floor Classes)")
    with col_pc:
        if per_class_path.exists():
            st.markdown("**Per-Class Official Test Metrics**")
            per_class_df = pd.read_csv(per_class_path)
            st.dataframe(per_class_df.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}",
            }), width="stretch", hide_index=True)

with tab4:
    results_path = OUTPUT_DIR / "architecture_comparison.csv"
    if results_path.exists():
        results = pd.read_csv(results_path).sort_values("macro_f1_mean", ascending=False)
        columns = [
            "configuration", "macro_f1_mean", "macro_f1_std", "accuracy_mean",
            "accuracy_std", "building_accuracy_mean", "building_accuracy_std",
        ]
        st.markdown(f"**5-Fold Person-Grouped CV ({len(results)} Setups, Default Settings)**")
        st.dataframe(
            results[columns].style.format({column: "{:.4f}" for column in columns[1:]}),
            width="stretch",
            hide_index=True,
        )
