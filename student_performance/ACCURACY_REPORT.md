# Accuracy Report — Student Performance Prediction

**Dataset:** UCI Student Performance (Mathematics), 395 students, 33 original attributes
**Date:** 2026-07-31
**Split:** 80% train (316) / 20% test (79), stratified, random_state=42

---

## 1. Data Preparation

| Step | Result |
|---|---|
| Raw rows loaded | 397 |
| Exact duplicates removed | 2 |
| Final rows | 395 |
| Missing values | 0 |
| Categorical features encoded | 17 (binary map + one-hot) |
| Features after encoding | 39 usable |

**File format note:** the distributed file carried a `.csv` extension but was an
XLSX workbook (verified via the ZIP/PK byte signature) and was read accordingly.

---

## 2. Targets

| Target | Type | Definition | Distribution |
|---|---|---|---|
| `G3` | Regression | Final grade, 0–20 | mean 10.42, sd 4.58 |
| `passed` | Classification | 1 if G3 ≥ 10 | 265 pass (67.1%) / 130 fail (32.9%) |

38 students (9.6%) have G3 = 0. All 38 also have `absences` = 0, which indicates
administrative deregistration rather than perfect attendance. These rows were
retained; filtering by the target would introduce selection bias.

---

## 3. Regression — Linear Regression (target: G3)

| Feature set | n | R² train | R² test | RMSE | MAE | CV R² (5-fold) |
|---|---|---|---|---|---|---|
| All features (no G1/G2) | 39 | 0.289 | 0.141 | 4.20 | 3.40 | 0.001 |
| Significant only (p<.05) | 12 | 0.201 | 0.162 | 4.15 | 3.28 | 0.122 |
| Including G1/G2 (leakage) | 41 | 0.866 | 0.724 | 2.38 | 1.65 | 0.784 |

**Selected model:** significant-features set. The 39-feature model reaches a
cross-validated R² of 0.001, i.e. no better than predicting the mean.
Reducing to 12 features raises CV R² to 0.122.

**Sensitivity (dropouts excluded):** R² rises from 0.162 to 0.266 (n = 357).
Reported for transparency; the full dataset remains the primary analysis.

**Practical accuracy:** RMSE 4.15 on a 0–20 scale, against a target
standard deviation of 4.58 — roughly 9% better than a mean-prediction baseline.
Not usable for individual decisions.

---

## 4. Classification — Decision Tree (target: passed)

### 4.1 Scenario comparison

| Scenario | Predictors | Accuracy | Balanced Acc. | F1 macro | Recall (Fail) | ROC AUC |
|---|---|---|---|---|---|---|
| C. Context only | 39 | 0.646 | 0.579 | 0.581 | 0.385 | 0.588 |
| B. Context + G1 | 40 | 0.797 | 0.839 | 0.792 | 0.962 | 0.897 |
| A. Context + G1 + G2 | 41 | 0.873 | 0.896 | 0.866 | 0.962 | 0.941 |
| Majority baseline | — | 0.671 | 0.500 | 0.402 | 0.000 | 0.500 |

Scenarios follow Cortez & Silva (2008), reflecting the point in the school year
at which a prediction would be made. Scenario B is the operational one: it can be
run after the first term, while intervention is still possible.

### 4.2 Confusion matrix — Scenario B

|  | Predicted Fail | Predicted Pass |
|---|---|---|
| **Actual Fail** | 25 | 1 |
| **Actual Pass** | 15 | 38 |

Recall on the Fail class is 0.962: 25 of 26
at-risk students are identified. The 15 false positives are an
acceptable cost for an early-warning system, where a missed at-risk student is
more costly than an unnecessary tutoring referral.

### 4.3 Model reduction

The tuned tree assigns feature importance 1.0000 to `G1` and 0 to all others.
Its depth-2 splits are redundant (both branches return the same class), so it
reduces to a single rule:

> **If G1 ≤ 10 → predicted to fail. Otherwise → predicted to pass.**

| Feature set | n | Test Balanced Acc. |
|---|---|---|
| G1 alone | 1 | 0.839 |
| G1 + failures | 2 | 0.839 |
| G1 + top-5 context | 6 | 0.839 |
| G1 + all context | 40 | 0.839 |

Predictions of the one-rule model and the full tree are **identical**. The 39
context features carry no incremental information once G1 is known.

---

## 5. Algorithm Comparison (balanced accuracy, test set)

| Model | Scenario C | Scenario B |
|---|---|---|
| One rule (G1 > 10) | — | **0.839** |
| Decision Tree | 0.579 | **0.839** |
| Random Forest | 0.579 | 0.829 |
| Logistic Regression | 0.533 | 0.733 |
| Gaussian Naive Bayes | 0.569 | 0.675 |
| KNN (k=15) | 0.529 | 0.606 |
| Majority baseline | 0.500 | 0.500 |

Five algorithmically distinct models converge near 0.55–0.58 under Scenario C.
Convergence at a low ceiling indicates the limitation lies in the data, not in
model choice. Naive Bayes underperforms because its independence assumption is
violated (Dalc↔Walc r=0.648; Medu↔Fedu r=0.623) and Gaussian likelihoods are a
poor fit for 1–5 ordinal scales. KNN degrades under 40 dimensions.

---

## 6. Conclusions

1. **Socioeconomic context does not predict individual academic performance.**
   With 39 context features, no model exceeded 0.579 balanced accuracy.
   The strongest single correlation with G3 was `failures` (r = −0.360).

2. **One prior grade outperforms 39 context variables.** Adding G1 raises
   balanced accuracy from 0.579 to 0.839. The resulting model uses G1 exclusively.

3. **Model complexity added nothing.** A single threshold matched Random Forest
   and exceeded Logistic Regression, Naive Bayes and KNN.

4. **Metric selection changes the model.** Tuning on `f1` (majority class only)
   selected a degenerate depth-1 tree. `balanced_accuracy` was used instead.

5. **Dropouts are a distinct population.** The 38 students with G3 = 0 follow a
   declining path (G1 7.53 → G2 4.66 → G3 0) but are indistinguishable from
   ordinary failing students at G1 (7.53 vs 7.71). Their divergence occurs in G2.

## 7. Limitations

- n = 395 from two Portuguese schools; not generalisable without revalidation.
- Cross-sectional data; no causal claims are made. `famsup` shows a negative
  coefficient, most plausibly reverse causation.
- The `absences` = 0 pattern among dropouts is an administrative artifact.
  Removing the feature changed balanced accuracy by 0.000 in Scenarios A and B.
- Self-reported variables (study time, alcohol use, going out) carry response bias.

## Reference

Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school
student performance. *Proceedings of 5th FUture BUsiness TEChnology Conference*.
