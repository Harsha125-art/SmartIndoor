# SmartIndoor: Improved Wi-Fi Fingerprinting-Based Indoor Localization Using Hybrid Feature Selection and Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 1. Project Introduction

Indoor localization remains a fundamental challenge in ubiquitous computing and context-aware systems, where satellite-based positioning systems (such as GPS/GNSS) fail due to signal attenuation, multi-path reflections, and severe non-line-of-sight (NLOS) conditions caused by structural walls and floors. 

**SmartIndoor** is a machine learning framework for Wi-Fi fingerprinting-based indoor localization developed on the benchmark **UJIIndoorLoc** dataset. It reformulates the localization problem into a multi-class hierarchical classification task covering 13 discrete campus building-floor pairs. By introducing a fold-fitted **Hybrid Feature Selection (Mutual Information + ANOVA F-score)** mechanism with correlation filtering and rigorous **Grouped Cross-Validation**, SmartIndoor achieves high classification performance (90.2% official validation accuracy, 0.884 macro F1) while preventing spatial and human-subject data leakage.

---

## 2. Problem Statement

Given a sparse, noisy vector of Received Signal Strength Indicator (RSSI) measurements from 520 Wireless Access Points (WAPs):
$$\mathbf{x} = [r_1, r_2, \dots, r_{520}] \in [-104, 0]^{520} \cup \{-100\}^{520}$$
where $-100\text{ dBm}$ denotes undetected access points, the objective is to predict the true architectural building and floor location:
$$y \in \{(\text{Building}, \text{Floor})\} = \{0\_0, 0\_1, 0\_2, 0\_3, 1\_0, 1\_1, 1\_2, 1\_3, 2\_0, 2\_1, 2\_2, 2\_3, 2\_4\}$$

Key challenges in this domain include:
1. **Severe High-Dimensional Sparsity:** In typical indoor spaces, only 2% to 6% of the 520 access points are detectable in any single reading; the vast majority are undetectable sentinels.
2. **Subject & Device Heterogeneity:** Signal propagation varies drastically depending on human body orientation, walking patterns, and phone hardware.
3. **Collinearity and Noise:** Neighboring access points often broadcast redundant, highly correlated RSSI signals, which can cause overfitting in linear models and degrade distance metrics in neighborhood-based classifiers.

---

## 3. Dataset Description and Source

This project benchmarks on the internationally recognized **UJIIndoorLoc** dataset:
- **Source:** [UCI Machine Learning Repository: UJIIndoorLoc Dataset](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
- **Reference:** J. Torres-Sospedra et al., *"UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems"*, IPIN 2014.
- **Coverage:** Universitat Jaume I campus, covering 3 multi-floor buildings across almost 110,000 m²:
  - **Building 0:** 4 floors (Floors 0–3)
  - **Building 1:** 4 floors (Floors 0–3)
  - **Building 2:** 5 floors (Floors 0–4)
  - **Total discrete locations:** 13 building-floor classes
- **Splits:**
  - **Training Split (`trainingData.csv`):** 19,937 reference fingerprints collected by 18 distinct survey users (`USERID` 1–18) across 25 different mobile devices (`PHONEID`).
  - **Official Validation Split (`validationData.csv`):** 1,111 testing fingerprints gathered separately to emulate real-world query localization.
- **Features:** 520 continuous WAP attributes (`WAP001` through `WAP520`), with metadata (`LONGITUDE`, `LATITUDE`, `FLOOR`, `BUILDINGID`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`).

---

## 4. Existing Project Limitations (Preserved Baseline)

The legacy baseline (accessible via `python main.py --baseline`) suffered from several methodological flaws common in early literature:
1. **Validation Split Discarded:** The original script discarded the official 1,111-row test partition (`DROP_VAL = True`) and instead performed a random 80/20 train/test holdout on training data.
2. **Data Leakage via Random Splitting:** Because measurements from the same user were randomly distributed between train and test sets, the model memorized user-specific walk paths rather than learning environmental Wi-Fi radio propagation.
3. **Global Feature Selection Leakage:** PCA and variance thresholding were executed prior to evaluation splits, leaking test-set statistics into the training pipeline.
4. **Ad-Hoc Signal Transformation:** The `-98 dBm` null floor and arbitrary `(1 - X / min(X))` scaling distorted RSSI logarithmic decay curves.
5. **Linear SVM Misalignment:** The baseline utilized a linear SVM for non-linear RSSI attenuation curves, leading to non-convergence warnings (`max_iter=1000`).

---

## 5. Our Proposed Improvements

SmartIndoor addresses each limitation through the following engineering and statistical advances:

| Dimension | Preserved 2019 Baseline | SmartIndoor (Proposed) |
| :--- | :--- | :--- |
| **Target Formulation** | Independent Building/Floor classifiers & Lat/Lon regressors | Unified 13-class joint Building-Floor target |
| **Validation Protocol** | Random 80/20 train holdout (`DROP_VAL=True`) | 3-Fold `StratifiedGroupKFold` on `USERID`, official validation set untouched until test |
| **Feature Isolation** | PCA fit across all rows before split | Fold-local fit; zero leakage into validation sets |
| **Feature Selection** | Simple variance threshold / PCA | MI, ANOVA F-Score, Hybrid Ranking, and Correlation Filtering |
| **Input Feature Hygiene** | Coordinates and metadata present in frames | Strict input isolation: only `WAP001`–`WAP520` used as model features |
| **Signal Normalization** | Arbitrary division by global minimum | Sentinel `100` mapped to `-100.0 dBm`, standard z-score scaling fit per fold |
| **Probability Calibration** | None (raw scores/decision functions) | Sigmoid Platt calibration with cross-validation for genuine confidence outputs |
| **User Interface** | Static matplotlib popups | Production Streamlit interactive web dashboard |
| **Faculty Integration** | Terminal only | Native MATLAB `Main.m` auto-launcher |

---

## 6. Hybrid Feature-Selection Methodology

Wi-Fi access points exhibit complex characteristics: some have linear floor-separation power, while others capture non-linear local multi-path fading. SmartIndoor evaluates four distinct feature selection strategies across four feature budgets ($K \in \{25, 50, 100, 150\}$):

1. **Mutual Information (MI):**
   Measures non-linear dependency between WAP RSSI and the building-floor class:
   $$I(X_j; Y) = \sum_{y \in Y} \int p(x_j, y) \log \frac{p(x_j, y)}{p(x_j)p(y)} dx_j$$
2. **ANOVA F-Score:**
   Evaluates linear variance between class means relative to within-class variance:
   $$F_j = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$$
3. **Hybrid Rank Combination (MI + ANOVA):**
   Overcomes individual ranking biases by assigning average fractional ranks to both metrics and minimizing their combined rank:
   $$R_{\text{hybrid}}(X_j) = \text{Rank}(-I(X_j; Y)) + \text{Rank}(-F_j)$$
4. **Correlation-Filtered Hybrid Ranking:**
   Iterates through the hybrid-ranked candidates and greedily rejects access points that exhibit Pearson correlation coefficient $|\rho| \ge 0.95$ with any higher-ranked AP already selected. This eliminates redundant colocated beacons.

All scoring and ranking operations are computed **strictly inside each cross-validation training fold**.

---

## 7. Machine Learning Algorithms

Four distinct classifier families are evaluated across all feature subsets:
1. **Support Vector Machine (RBF Kernel):**
   Maps RSSI fingerprints to an infinite-dimensional Hilbert space to construct optimal maximum-margin hyperplanes ($C=10.0$, $\gamma=\text{scale}$).
2. **Linear Discriminant Analysis (LDA with Shrinkage):**
   Projects features onto directions that maximize class separability. Employs Ledoit-Wolf automatic covariance shrinkage (`shrinkage="auto"`, `solver="lsqr"`) to handle high-dimensional collinearity.
3. **K-Nearest Neighbors (KNN):**
   Memory-based instance classifier using distance weighting ($k=5$, weights inversely proportional to Euclidean distance).
4. **Random Forest (RF):**
   Ensemble of 160 randomized decision trees (`min_samples_leaf=1`, `random_state=42`) providing non-linear bagging robustness.

---

## 8. Experimental Methodology & Rationale for 3 Folds

### Grouped Cross-Validation (`StratifiedGroupKFold`)
In real-world deployment, indoor localization systems are queried by unseen users. To guarantee genuine out-of-subject generalization, cross-validation groups samples by `USERID`. 

### Why 3 Folds?
The UJIIndoorLoc training set contains measurements from 18 unique surveyors (`USERID` 1–18). Users surveyed specific corridors and buildings rather than the entire campus. 
- With 5 or 10 folds, partitioning 18 users into test folds inevitably leaves at least one training fold missing entire building-floor classes (e.g., no samples for Building 2 Floor 4).
- With **3 folds**, each fold retains approximately 12 users in training, ensuring 100% representation across all 13 building-floor classes while maintaining strict user-group separation.

### Double-Blind Final Evaluation
The official validation split (`validationData.csv`, 1,111 rows) is kept strictly isolated in storage until all 64 model configurations are cross-validated. Only the best model is fitted on training data, calibrated, and evaluated once on the validation dataset.

---

## 9. Experimental Results and Analysis

### Official Untouched Validation Performance
Evaluated on the official 1,111 validation samples:

| Evaluation Metric | Measured Result |
| :--- | :---: |
| **Accuracy** | **90.19%** (1,002 / 1,111) |
| **Macro F1-Score** | **0.8840** |
| **Macro Precision** | **0.8808** |
| **Macro Recall** | **0.8949** |

### Grouped Cross-Validation: Best Model by Architecture

| Model | Winning Feature Selection | Features ($K$) | Grouped-CV Accuracy | Grouped-CV Macro F1 |
| :--- | :--- | :---: | :---: | :---: |
| **SVM (RBF)** | Mutual Information | 150 | **0.8660** | **0.8519** |
| **LDA** | Correlation-Filtered Hybrid | 150 | 0.8556 | 0.8436 |
| **KNN** | Mutual Information | 150 | 0.8507 | 0.8393 |
| **Random Forest** | Correlation-Filtered Hybrid | 150 | 0.8420 | 0.8268 |

### Grouped Cross-Validation: Best Configuration per Feature Selection Strategy

| Feature Selection Strategy | Winning Model | Features ($K$) | Grouped-CV Macro F1 |
| :--- | :--- | :---: | :---: |
| **Mutual Information** | SVM (RBF) | 150 | **0.8519** |
| **Correlation-Filtered Hybrid** | LDA | 150 | 0.8436 |
| **Hybrid (MI + ANOVA)** | SVM (RBF) | 150 | 0.8344 |
| **ANOVA F-Score** | SVM (RBF) | 150 | 0.8203 |

All 64 experimental permutations are saved in `output/smartindoor/cross_validation_results.csv`.

---

## 10. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- (Optional) MATLAB R2020a or newer

### Setup Virtual Environment

```powershell
# Clone or navigate to the project directory
cd Localization_via_WiFi_Fingerprinting-master

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 11. How to Run the Project

### 1. Run Complete Machine Learning Experiment
To run all 64 feature-selection/model permutations across 3-fold grouped CV and evaluate on the official validation set:
```powershell
python main.py
```
*(To run faster with custom folds, use `python main.py --folds 2`)*

### 2. Launch from MATLAB
Execute `Main.m` directly in MATLAB:
```matlab
>> Main
```
The script will detect your Python environment, execute `main.py`, and stream output to the MATLAB Command Window.

### 3. Launch Interactive Streamlit Dashboard
```powershell
streamlit run smartindoor/dashboard.py
```
Open your browser at `http://localhost:8501` to test Wi-Fi fingerprints via JSON or CSV.

### 4. Run Preserved 2019 Baseline
To reproduce the legacy baseline for historical comparison:
```powershell
python main.py --baseline
```

---

## 12. Dashboard Overview

The **SmartIndoor** dashboard provides:
1. **Interactive Fingerprint Input:** Accepts direct JSON payloads (e.g. `{"WAP011": -60, "WAP012": -61}`) or uploaded single-row CSV files.
2. **Location Prediction:** Displays predicted Building and Floor with large status metrics.
3. **Calibrated Confidence:** Displays posterior probability generated by the sigmoid-calibrated SVM.
4. **Experimental Visualizations:**
   - Model comparison bar chart with score annotations.
   - Feature selection scaling curve comparing MI, ANOVA, and Hybrid strategies.
   - Annotated 13×13 confusion matrix on the official validation dataset.
   - Interactive cross-validation leaderboard table.

---

## 13. Team Contributions

| Team Member | Primary Roles & Responsibilities |
| :--- | :--- |
| **Lead ML Engineer** | Architecture design, scikit-learn pipeline construction, probability calibration, and leak-free CV splitting. |
| **Data & Feature Specialist** | UJIIndoorLoc preprocessing, WAP sentinel normalization, MI & ANOVA hybrid ranking, and correlation filtering. |
| **UI & Systems Developer** | Streamlit dashboard design, MATLAB `Main.m` runner, chart styling, and input validation. |
| **Research & Documentation** | Experimental audit, metrics verification, literature review, and academic report compilation. |

---

## 14. Repository Structure

```text
Localization_via_WiFi_Fingerprinting-master/
├── data/
│   ├── trainingData.csv                # 19,937 UJIIndoorLoc training fingerprints
│   └── validationData.csv              # 1,111 official validation fingerprints
├── output/
│   └── smartindoor/
│       ├── best_model.joblib           # Trained calibrated SVM model
│       ├── confusion_matrix.csv        # 13x13 numerical confusion matrix
│       ├── confusion_matrix.png        # Publication-quality annotated matrix plot
│       ├── cross_validation_results.csv# All 64 experimental CV scores
│       ├── feature_selection_comparison.png # Strategy scaling curves
│       ├── model_comparison.png        # Model performance bar chart
│       ├── model_metadata.json         # Feature names, model params, validation scores
│       ├── official_validation_metrics.csv # Overall official test scores
│       └── per_class_metrics.csv       # Class-level precision, recall, F1, support
├── scripts/                            # Preserved 2019 baseline source files
│   ├── errors.py
│   ├── models.py
│   ├── plots.py
│   └── utils.py
├── smartindoor/                        # SmartIndoor production package
│   ├── __init__.py
│   ├── dashboard.py                    # Streamlit web interface
│   ├── data.py                         # Leakage-safe data loader & target creator
│   ├── experiment.py                   # 64-config CV and validation runner
│   ├── feature_selection.py            # MI, ANOVA, Hybrid, and Correlation Selector
│   └── models.py                       # Model pipelines & classifier factories
├── Main.m                              # Faculty MATLAB entry point
├── main.py                             # Main command-line experiment launcher
├── requirements.txt                    # Python package dependencies
└── README.md                           # Complete academic project documentation
```

---

## 15. Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If using this implementation in academic work, please cite:
```bibtex
@misc{smartindoor2026,
  title={SmartIndoor: Improved Wi-Fi Fingerprinting-Based Indoor Localization Using Hybrid Feature Selection and Machine Learning},
  author={SmartIndoor Project Team},
  year={2026},
  url={https://github.com/Harsha125-art/SmartIndoor.git}
}
```
