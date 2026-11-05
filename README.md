# SmartIndoor: Improved Wi-Fi Fingerprinting-Based Indoor Localization Using Hybrid Feature Selection and Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 1. Project Summary

GPS is unreliable indoors because walls and floors block and reflect its signals. **SmartIndoor** predicts which building and floor a phone is on using only the Wi-Fi signals it detects, and also estimates its coordinates. It uses the public **UJIIndoorLoc** dataset and classical pattern recognition methods. No deep learning is used.

The final model is an **RBF SVM with correlation-filtered hybrid feature selection**, which uses 200 of the 520 access points:

| Official test file (1,111 rows) | Result |
| :--- | :---: |
| **Accuracy** | **90.55%** (1,006 / 1,111) |
| **Macro F1** | **0.8866** |
| **Macro precision / recall** | 0.8851 / 0.9021 |
| **Correct building** | 99.73% (3 errors) |
| **Mean coordinate error, all rows** | **8.92 m** |
| **Mean coordinate error, correct building and floor** | 7.80 m |

**5-fold person-grouped CV** on the training file gives macro F1 0.851 ± 0.100 and accuracy 88.5% ± 6.5%. These are optimistic because the settings were tuned on the same folds. On three other groupings of people, macro F1 is 0.775 ± 0.147 and accuracy is 81.4% ± 11.7%.

**How to read these results:** the final model clearly beats an SVM with no feature selection, and uses 62% fewer access points than plain 1-nearest neighbor while performing about the same. It does not clearly beat 1-nearest neighbor. See [Section 7](#7-results).

---

## 2. Problem Statement

Each sample is a vector of 520 received signal strengths (RSSI), one per wireless access point, in dBm. In the dataset files, `+100` means the access point was not detected. A scan detects about 18 of the 520 access points on average, so the data is very sparse.

The classification target is one of 13 building-floor classes, written `building_floor`:

`0_0, 0_1, 0_2, 0_3, 1_0, 1_1, 1_2, 1_3, 2_0, 2_1, 2_2, 2_3, 2_4`

Accuracy depends on more than the building's layout, because phones report different signal levels and each person carries a phone differently.

---

## 3. Dataset

**UJIIndoorLoc**, from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc), was collected across three buildings at Universitat Jaume I.

| | Training file | Official test file (`validationData.csv`) |
| :--- | :---: | :---: |
| **Rows** | 19,937 | 1,111 |
| **People** | 18, identified by `USERID` | 14 per the paper; not identified (`USERID` 0) |
| **Devices** | 16 | 11, of which 2 also appear in training |
| **Locations** | Fixed reference points | Arbitrary points |
| **Timing** | First | Captured months later |

Each row has 520 signal features plus longitude, latitude, floor, building, and other metadata. Only the 520 signal features are used as model inputs. Coordinates are UTM values in meters.

Reference: J. Torres-Sospedra et al., ["UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems"](https://doi.org/10.1109/IPIN.2014.7275492), IPIN 2014.

---

## 4. Evaluation Method

- **Model comparison:** 5-fold CV with `StratifiedGroupKFold` on the training file, grouped by `USERID`. No person appears on both sides of a fold, and stratification is as close as that grouping allows.
- **Final test:** the official file is not used to choose or tune any model.
- **No leakage:** imputation, scaling, and feature selection are fitted inside each training fold.
- **Reported metrics:** accuracy, macro F1, mean ± SD over folds, confusion matrix, and per-class precision and recall.

Folds differ in difficulty and may lack some classes, so macro F1 for each fold covers the classes it contains. The SDs are large partly for this reason. Only users 1 and 11 surveyed Building 0, so results depend heavily on which people are held out.

---

## 5. Methods

### Feature selection

1. **Mutual Information (MI):** how much an access point's signal reveals about the class, including non-linear relationships.
2. **ANOVA F-score:** how much an access point's mean signal differs between classes.
3. **Hybrid:** adds each access point's MI rank and ANOVA rank, and prefers the lowest total.
4. **Correlation-filtered hybrid:** goes through the hybrid ranking and rejects any access point with absolute Pearson correlation of 0.95 or more with one already kept.

### Models compared

- **RBF SVM**, with and without each feature selection method.
- **Gradient boosting** (`HistGradientBoostingClassifier`).
- **Two-stage classifiers**, which predict the building and then use that building's own floor model and access points.
- **Relative signal input**, which expresses each signal relative to the strongest in its scan, to reduce differences between phones.
- **1-nearest neighbor**, the method from the UJIIndoorLoc paper, as a reference.

### Tuning

Three finalists were tuned on the training file. Their settings were chosen on the 5-fold split and rechecked on three other groupings of people, because the best score on the tuning folds is optimistic. All 18 people appear in every grouping, so this reduces the optimism but does not remove it.

| Finalist | Settings searched |
| :--- | :--- |
| SVM, filtered hybrid or ANOVA | `C`: 1, 10, 100; `gamma`: scale, 0.001, 0.003, 0.01; access points: 100, 150, 200, 300 |
| Two-stage gradient boosting with relative input | Learning rate and rounds: 0.1/100, 0.05/200, 0.1/200; leaves: 15, 31; access points: 100, 150, 300 |

### Final model choice

The selection rule set in advance took the simplest setup within one SD of the best CV score, which picked SVM with ANOVA. After tuning, the three finalists were effectively tied (macro F1 0.778, 0.775, and 0.768 on the other groupings). **SVM with filtered hybrid was chosen over the rule's pick** because it costs no measurable performance and identifies the building more reliably on those groupings (0.985 vs 0.975). This choice used only the training file.

Final settings: `C=1`, `gamma=0.001`, 200 access points, sigmoid calibration with person-grouped folds.

### Coordinates

The coordinate model was chosen by lowest mean error on the 5-fold split: **KNN with k=3, distance-weighted**, using the same 200 access points. Alternatives were KNN with other k values, random forest, and SVR, each fitted for the whole campus, per building, or per floor.

---

## 6. How to Run

### From MATLAB

```matlab
>> Main
```

`Main.m` runs everything without intervention and displays the results. The first run creates a Python environment in `.venv` and installs the packages, which needs Python 3.10 or newer and an internet connection. Set the `PYTHON_EXECUTABLE` environment variable to use a specific Python.

### From a terminal

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

| Command | What it does |
| :--- | :--- |
| `python main.py` | Runs the full pipeline, including the official test file |
| `python main.py --no-test` | Runs the CV stages only, without reading the test file |
| `python main.py --tune` | Also repeats the finalist tuning, about 40 minutes more |
| `python main.py --original` | Runs the original 64-setup 3-fold experiment |
| `python main.py --baseline` | Runs the 2019 baseline |
| `python -m smartindoor.clustering` | Runs the separate clustering check on the training file |
| `streamlit run smartindoor/dashboard.py` | Opens the dashboard |

The dashboard takes a fingerprint as JSON or a one-row CSV, then predicts the building and floor with a confidence score. It classifies supplied signals and does not scan live Wi-Fi. The saved model was built with scikit-learn 1.9.1, so if it does not load with another version, `python main.py` rebuilds it.

---

## 7. Results

### Official test file

| Method | Accuracy | Macro F1 | Building errors | Mean error, all rows | Mean error, correct location |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Final model** | **90.55%** | **0.887** | **3** | **8.92 m** | **7.80 m** |
| 1-nearest neighbor, our rerun of the paper's method | 90.91% | 0.894 | 4 | 9.20 m | 7.63 m |
| Original SmartIndoor model (SVM, MI, 150 access points) | 90.19% | 0.884 | 14 | Not predicted | Not predicted |
| UJIIndoorLoc paper, as published | 89.92% | Not reported | 3 | Not reported | 7.9 m |

The differences in accuracy amount to a few rows, within about ±1.7 percentage points of uncertainty for a test this size, so **these methods are effectively tied on the test file**. The final model was already fixed when 1-nearest neighbor was scored, and it was not changed afterward.

The rerun of the paper's method treats "not detected" as -100, and does not copy the paper's voting step for ties, so it does not match the published result exactly. Using `+100` as in the files gives 75.34%.

### Person-grouped CV

Macro F1 and accuracy, mean ± SD:

| Setup | Stratified 5-fold F1 | Stratified 5-fold accuracy | Other groupings F1 | Other groupings accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **SVM, filtered hybrid, tuned (final)** | **0.851 ± 0.100*** | **0.885 ± 0.065*** | **0.775 ± 0.147** | **0.814 ± 0.117** |
| 1-nearest neighbor (paper's method) | 0.842 ± 0.103 | 0.872 ± 0.073 | 0.764 ± 0.167 | 0.787 ± 0.165 |
| SVM, MI (original) | 0.780 ± 0.051 | 0.849 ± 0.045 | 0.737 ± 0.149 | 0.778 ± 0.148 |
| SVM, all access points (baseline) | 0.724 ± 0.123 | 0.727 ± 0.119 | 0.655 ± 0.119 | 0.655 ± 0.142 |

\* Optimistic, because the settings were tuned on these folds.

Fold-by-fold macro F1 difference for the final model:

| Compared with | Stratified 5-fold | Other groupings (15 folds) |
| :--- | :---: | :---: |
| SVM, all access points | +0.127, higher in 5 of 5 | +0.120, higher in 14 of 15 |
| SVM, MI (original) | +0.071, higher in 4 of 5 | +0.038, higher in 11 of 15 |
| 1-nearest neighbor | +0.009, higher in 3 of 5 | +0.011, higher in 10 of 15 |

Before tuning, on the stratified folds, the best setups were two-stage gradient boosting with relative input (0.840 ± 0.096), SVM with filtered hybrid (0.804 ± 0.071), and SVM with MI and relative input (0.793 ± 0.079). All 18 setups are in `output/smartindoor/architecture_comparison.csv`.

### Final model per class, official test file

| Class | Precision | Recall | F1 | Rows |
| :--- | :---: | :---: | :---: | :---: |
| `0_0` | 0.941 | 0.821 | 0.877 | 78 |
| `0_1` | 0.936 | 0.981 | 0.958 | 208 |
| `0_2` | 0.958 | 0.964 | 0.961 | 165 |
| `0_3` | 1.000 | 0.976 | 0.988 | 85 |
| `1_0` | 0.574 | 0.900 | 0.701 | 30 |
| `1_1` | 0.980 | 0.671 | 0.797 | 143 |
| `1_2` | 0.741 | 0.954 | 0.834 | 87 |
| `1_3` | 0.896 | 0.915 | 0.905 | 47 |
| `2_0` | 0.846 | 0.917 | 0.880 | 24 |
| `2_1` | 0.939 | 0.973 | 0.956 | 111 |
| `2_2` | 0.957 | 0.833 | 0.891 | 54 |
| `2_3` | 0.841 | 0.925 | 0.881 | 40 |
| `2_4` | 0.897 | 0.897 | 0.897 | 39 |

Of the 105 errors, 102 are the correct building but the wrong floor. Building 1 is the weakest. The confusion matrix is in `output/smartindoor/final/confusion_matrix.png`.

### Coordinate error

| | Mean, all rows | Median, all rows | 95th percentile | Mean, correct location |
| :--- | :---: | :---: | :---: | :---: |
| **Official test file** | **8.92 m** | 5.93 m | 26.67 m | 7.80 m |

In CV, mean error over all rows is 11.04 ± 3.05 m on the stratified folds and 12.37 ± 4.89 m on the other groupings. On the test file, mean error is 6.37 m in Building 0, 10.56 m in Building 1, and 12.15 m in Building 2.

### Clustering check

K-means clustering on the training file, without labels, recovers the buildings better than the 13 classes. Adjusted Rand index (ARI) is 0.62 to 0.63 for buildings with 3 clusters and 0.27 to 0.32 for classes with 13 clusters. Within each building, ARI for floors is 0.27 in Building 0, 0.05 in Building 1, and 0.29 in Building 2. The silhouette score has no peak at 13 clusters, rising from 0.21 there to 0.27 at 30, the most tested. With 13 clusters, normalized mutual information is higher with users (0.42) than with floors (0.26). Results are in `output/smartindoor/final/clustering_*.csv`.

---

## 8. Limitations

- **The test file's people are unidentified.** The paper reports that user IDs were not recorded in that phase, and phones 13 and 14 appear in both phases. So the test score cannot be described as performance on proven new people. The person-grouped CV scores address that question.
- **Earlier use of the test file.** The original 90.19% model had already been scored on this file before the final model was developed, and one early comparison used it as development data. Its scores were discarded, and the final settings were chosen using only the training file.
- **Some test coordinates may be predictions.** When a user confirmed a location shown by the collection app, the app stored that predicted location. Test coordinate errors may therefore be somewhat optimistic.
- **Few people per floor.** Building 0 was surveyed by only two people in the training file.
- **Confidence runs low.** Mean confidence on the test file is 80%, against 90.55% accuracy.
- **Tuning is not fully independent.** The same 18 people appear in all CV groupings.

---

## 9. Repository Structure

```text
SmartIndoor/
├── data/                       # UJIIndoorLoc training and official test files
├── output/
│   ├── smartindoor/
│   │   ├── architecture_comparison*.csv   # 5-fold CV for all 18 setups
│   │   ├── tuning_*.csv                   # Tuning grid and recheck
│   │   ├── final/                         # Final model, test results, coordinates
│   │   └── ...                            # Original 64-setup experiment results
│   └── ...                     # 2019 baseline results
├── scripts/                    # 2019 baseline code
├── smartindoor/
│   ├── data.py                 # Data loading and target creation
│   ├── feature_selection.py    # MI, ANOVA, hybrid, correlation filtering
│   ├── models.py               # Classifiers, relative input, two-stage model
│   ├── compare.py              # 5-fold person-grouped comparison
│   ├── tune.py                 # Finalist tuning and recheck
│   ├── final.py                # Final model and test evaluation
│   ├── coordinates.py          # Coordinate prediction and paper's method
│   ├── clustering.py           # Unsupervised check of the classes
│   ├── experiment.py           # Original 64-setup experiment
│   └── dashboard.py            # Streamlit dashboard
├── Main.m                      # MATLAB entry point
├── main.py                     # Command-line entry point
└── requirements.txt
```

---

## 10. Credits and License

This project builds on baseline code by Ryan Clark, Matt Hong, and So Sosaki (2019), released under the MIT License. See [LICENSE](LICENSE). The dataset and the 1-nearest neighbor benchmark are from Torres-Sospedra et al. (2014).

### Team contributions

| Role | Responsibilities |
| :--- | :--- |
| **Lead ML Engineer** | Architecture design, scikit-learn pipeline construction, probability calibration, and leak-free CV splitting. |
| **Data & Feature Specialist** | UJIIndoorLoc preprocessing, WAP sentinel normalization, MI & ANOVA hybrid ranking, and correlation filtering. |
| **UI & Systems Developer** | Streamlit dashboard design, MATLAB `Main.m` runner, chart styling, and input validation. |
| **Research & Documentation** | Experimental audit, metrics verification, literature review, and academic report compilation. |

To cite this project:

```bibtex
@misc{smartindoor2026,
  title={SmartIndoor: Improved Wi-Fi Fingerprinting-Based Indoor Localization Using Hybrid Feature Selection and Machine Learning},
  author={SmartIndoor Project Team},
  year={2026},
  url={https://github.com/Harsha125-art/SmartIndoor.git}
}
```
