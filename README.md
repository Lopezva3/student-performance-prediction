# Student Performance Prediction

<div align="center">

**Can socioeconomic context predict academic success?**
*A machine learning analysis of 395 secondary school students*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.0-150458?style=flat-square&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.14-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

**Author:** Valeria López

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Valeria_López-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/valeria-l%C3%B3pez-44186035b)

</div>

---

## The Short Answer

> **No.**
>
> With **39** socioeconomic and lifestyle features, no algorithm exceeded **0.579** balanced accuracy — barely above random guessing.
>
> Add **one** prior grade, and accuracy jumps to **0.839**. But here is the twist: the resulting model uses that single grade **exclusively**, collapsing into one rule:
>
> ### `if G1 ≤ 10 → predicted to fail`
>
> This one-line rule produces **predictions identical** to a 40-feature Decision Tree, and beats Random Forest, Logistic Regression, Naive Bayes and KNN.

---

## Table of Contents

- [Objective](#objective)
- [Dataset](#dataset)
- [Results](#results)
- [The Core Finding](#the-core-finding)
- [Methodology Notes](#methodology-notes)
- [Repository Structure](#repository-structure)
- [Reproduction](#reproduction)
- [Limitations](#limitations)
- [Reference](#reference)

---

## Objective

Predict secondary school academic performance using two required models:

| Model | Task | Target |
|:--|:--|:--|
| **Linear Regression** | Regression | `G3` — final grade (0–20) |
| **Decision Tree Classifier** | Classification | `passed` — pass/fail (G3 ≥ 10) |

Three additional algorithms were benchmarked to test whether model choice, rather than data quality, was the limiting factor.

---

## Dataset

**UCI Student Performance** — Mathematics course, two Portuguese secondary schools.

| | |
|:--|:--|
| **Students** | 395 (after removing 2 exact duplicates) |
| **Original attributes** | 33 |
| **Usable features after encoding** | 39 |
| **Missing values** | 0 |
| **Class balance** | 265 pass (67.1%) / 130 fail (32.9%) |
| **Split** | 80/20 stratified, `random_state=42` |

**Feature groups:** demographics · family background · school factors · lifestyle · grades

---

## Results

### Classification — Decision Tree

Scenarios follow **Cortez & Silva (2008)**. Each represents a different point in the school year at which a prediction could realistically be made.

| Scenario | Predictors | Accuracy | Balanced Acc. | Recall (Fail) | ROC AUC |
|:--|--:|--:|--:|--:|--:|
| **C.** Context only | 39 | 0.646 | 0.579 | 0.385 | 0.588 |
| **B.** Context + G1 | 40 | 0.797 | **0.839** | **0.962** | **0.897** |
| **A.** Context + G1 + G2 | 41 | 0.873 | 0.896 | 0.962 | 0.941 |
| — Majority baseline | — | 0.671 | 0.500 | 0.000 | 0.500 |

**Scenario B is the operational model.** It runs after the first term, while intervention is still possible.

#### Confusion Matrix — Scenario B

| | Predicted Fail | Predicted Pass |
|:--|--:|--:|
| **Actual Fail** | **25** | 1 |
| **Actual Pass** | 15 | 38 |

**25 of 26** at-risk students identified. The 15 false positives are an acceptable trade-off: an unnecessary tutoring referral costs far less than a missed at-risk student.

---

### Regression — Linear Regression

| Feature set | n | R² train | R² test | RMSE | CV R² |
|:--|--:|--:|--:|--:|--:|
| All features (no G1/G2) | 39 | 0.289 | 0.141 | 4.20 | **0.001** |
| Significant only (p < .05) | 12 | 0.201 | 0.162 | 4.15 | **0.122** |
| *Including G1/G2 (leakage)* | 41 | *0.866* | *0.724* | *2.38* | *0.784* |

**Fewer features, better model.** The 39-feature version has a cross-validated R² of 0.001 — statistically indistinguishable from predicting the mean. With 316 training rows and 39 features, it memorises noise.

RMSE of 4.15 against a target standard deviation of 4.58 means roughly **9% improvement over a mean-prediction baseline**. Not usable for individual decisions.

---

### Algorithm Comparison

Balanced accuracy, test set:

| Model | Scenario C | Scenario B |
|:--|--:|--:|
| **One rule** (`G1 > 10`) | — | **0.839** |
| Decision Tree | 0.579 | **0.839** |
| Random Forest | 0.579 | 0.829 |
| Logistic Regression | 0.533 | 0.733 |
| Gaussian Naive Bayes | 0.569 | 0.675 |
| KNN (k=15) | 0.529 | 0.606 |
| Majority baseline | 0.500 | 0.500 |

Five algorithmically distinct models converge near **0.55–0.58** under Scenario C. When approaches this different hit the same ceiling, the limitation lies in the **data**, not in model selection.

*Naive Bayes underperforms because its independence assumption is violated (`Dalc`↔`Walc` r = 0.648; `Medu`↔`Fedu` r = 0.623), and Gaussian likelihoods poorly fit 1–5 ordinal scales. KNN degrades under 40 dimensions.*

---

## The Core Finding

The tuned Decision Tree assigns feature importance **1.0000** to `G1` and **0** to all 39 others. Its depth-2 splits are redundant — both branches return the same class.

| Feature set | n | Test Balanced Acc. |
|:--|--:|--:|
| G1 alone | 1 | 0.839 |
| G1 + failures | 2 | 0.839 |
| G1 + top-5 context | 6 | 0.839 |
| G1 + all context | 40 | 0.839 |

**Identical predictions across all four.** The 39 context features carry **zero** incremental information once the first-term grade is known.

### What this means in practice

A school does not need machine learning or socioeconomic surveys to identify at-risk students.

**It needs to look at first-term grades.**

---

## Methodology Notes

Findings that shaped the analysis:

<details>
<summary><b>The file was not a CSV</b></summary>

<br>

The distributed file carried a `.csv` extension but failed to load with any encoding. Reading the raw bytes revealed the `PK\x03\x04` signature — the ZIP header — and an internal `xl/` path. It was an **XLSX workbook with a renamed extension**, and was read with `pd.read_excel()` accordingly.

</details>

<details>
<summary><b>The IQR rule fails on ordinal scales</b></summary>

<br>

The IQR method flagged **83 outliers** in `failures` because 79% of students have a value of 0, collapsing Q1 = Q3 = 0 and IQR = 0. Any non-zero value becomes an "outlier." The same distortion affects `studytime`, `famrel`, `freetime` and `Dalc` — all 1–5 Likert scales.

Conversely, it flagged **none** of the 38 students with G3 = 0, despite the histogram showing them as a visually isolated cluster. **Statistical rules alone are insufficient; domain knowledge is required.**

</details>

<details>
<summary><b>The scoring metric changes the model</b></summary>

<br>

Tuning on `f1` — which in scikit-learn measures only the positive (majority) class — selected a degenerate `max_depth=1` tree that matched the baseline exactly. Re-tuning on `balanced_accuracy` produced a usable model.

| Scoring metric | Selected depth |
|:--|--:|
| `f1` | 1 |
| `f1_macro` | 1 |
| `balanced_accuracy` | 3 |
| `roc_auc` | 8 |

Metric selection is a modelling decision, not a technical detail.

</details>

<details>
<summary><b>An administrative artifact, investigated and dismissed</b></summary>

<br>

All **38 of 38** students with G3 = 0 also have **exactly 0 recorded absences** — statistically impossible by chance. This indicates administrative deregistration rather than perfect attendance.

The feature was tested for leakage by retraining without it:

| Scenario | With `absences` | Without | Δ |
|:--|--:|--:|--:|
| C. Context only | 0.579 | 0.608 | −0.029 |
| B. Context + G1 | 0.839 | 0.839 | **+0.000** |
| A. Context + G1 + G2 | 0.896 | 0.896 | **+0.000** |

The model does **not** rely on the artifact. Documented, tested, dismissed.

</details>

<details>
<summary><b>Dropouts are a distinct population</b></summary>

<br>

| Group | G1 | G2 | G3 |
|:--|--:|--:|--:|
| Passed | 12.42 | 12.60 | 12.92 |
| Failed | 7.71 | 6.75 | 5.38 |
| **Dropouts** | **7.53** | **4.66** | **0.00** |

Dropouts and ordinary failing students are **indistinguishable at G1** (7.53 vs 7.71). They diverge only at G2. Disengagement is gradual, not sudden — but it cannot be detected from the first term alone.

These 38 rows were **retained**. Filtering rows by the target variable introduces selection bias.

</details>

<details>
<summary><b>Leakage was quantified, not hidden</b></summary>

<br>

`G2` correlates with `G3` at **r = 0.905**. Including it inflates regression R² from 0.162 to 0.724 — a 4.5× increase — while teaching nothing about the student.

Rather than silently excluding these variables, they are reported as **separate scenarios** with the inflation measured explicitly.

</details>

---

## Repository Structure

```
student-performance-prediction/
│
├── README.md                                  This file
├── Data_Visualization_and_Machine_Learning_
│   Fundamentals.ipynb                         Full analysis notebook
│
└── student_performance/
    │
    ├── ACCURACY_REPORT.md                     Detailed metrics report
    ├── metrics.json                           Machine-readable results
    ├── requirements.txt                       Dependencies
    │
    ├── data/
    │   └── student_math_clean.csv             Cleaned dataset (395 rows)
    │
    └── figures/
        ├── target_analysis.png                Target distribution & class balance
        ├── correlation_heatmap.png            Feature correlation matrix
        ├── regression_diagnostics.png         Residuals, coefficients, fit
        ├── decision_tree.png                  Trained tree visualisation
        ├── classifier_evaluation.png          Confusion matrices & ROC curves
        ├── one_rule_analysis.png              Single-threshold justification
        │
        ├── interactive_g1_vs_g3.html          Plotly — hover for student profiles
        ├── interactive_model_comparison.html  Plotly — benchmark chart
        └── interactive_trajectory.html        Plotly — grade paths by outcome
```
---

## Reproduction

```bash
git clone https://github.com/Lopezva3/student-performance-prediction.git
cd student-performance-prediction

pip install -r requirements.txt
jupyter notebook Data_Visualization_and_Machine_Learning_Fundamentals.ipynb
```

All results use `random_state=42` and an 80/20 stratified split. Cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True)`.

**Dependencies:** `pandas` · `numpy` · `matplotlib` · `seaborn` · `plotly` · `scikit-learn` · `openpyxl`

---

## Limitations

| | |
|:--|:--|
| **Sample** | n = 395 from two Portuguese schools. Not generalisable without revalidation on other populations. |
| **Causality** | Cross-sectional data supports no causal claims. `famsup` (family support) shows a *negative* coefficient — most plausibly reverse causation: families engage **because** grades are poor. |
| **Self-report bias** | Study time, alcohol consumption and social activity are self-reported and subject to social desirability bias. |
| **Estimator limits** | Mutual information returned 0.0000 for most binary features — a sample-size limitation, not evidence of irrelevance. |
| **Grade scale** | The 0–20 Portuguese scale with a pass mark of 10 does not transfer directly to other systems. |

---

## Reference

> Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In A. Brito & J. Teixeira (Eds.), *Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)* (pp. 5–12). EUROSIS.

Dataset available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).

---

<div align="center">

### Valeria López

[![LinkedIn](https://img.shields.io/badge/Connect_on_LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/valeria-l%C3%B3pez-44186035b)

*Data Visualization and Machine Learning Fundamentals* · 2026

</div>
