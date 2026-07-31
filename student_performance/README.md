# Student Performance Prediction

Machine learning analysis of secondary school student performance using the
UCI Student Performance dataset (Mathematics course, 395 students, Portugal).

## Objective

Predict final academic performance from student attributes, using:
- **Linear Regression** for the final grade (G3, scale 0-20)
- **Decision Tree Classifier** for pass/fail (G3 >= 10)

## Main Finding

Socioeconomic and lifestyle context does not predict individual performance.
With 39 context features, no algorithm exceeded **0.579 balanced accuracy**.

Adding a single prior grade (G1) raises this to **0.839** — and the resulting
model uses **G1 exclusively**, reducing to one rule:

> If G1 <= 10, the student is predicted to fail. Otherwise, to pass.

This single-threshold rule produces **predictions identical** to the full
40-feature Decision Tree, and outperforms Random Forest, Logistic Regression,
Naive Bayes and KNN.

## Results

### Classification (target: pass/fail)

| Scenario | Predictors | Balanced Acc. | Recall (Fail) | ROC AUC |
|---|---|---|---|---|
| C. Context only | 39 | 0.579 | 0.385 | 0.588 |
| B. Context + G1 | 40 | **0.839** | **0.962** | **0.897** |
| A. Context + G1 + G2 | 41 | 0.896 | 0.962 | 0.941 |
| Majority baseline | - | 0.500 | 0.000 | 0.500 |

Scenarios follow Cortez & Silva (2008): each reflects a different point in the
school year at which a prediction could be made. Scenario B is the operational
one — it runs after the first term, while intervention is still possible.

### Regression (target: G3)

| Feature set | R2 test | RMSE | CV R2 |
|---|---|---|---|
| All features (no G1/G2) | 0.141 | 4.20 | 0.001 |
| Significant only (p<.05) | 0.162 | 4.15 | 0.122 |

RMSE of 4.15 against a target standard deviation of 4.58: roughly 9% better
than predicting the mean. Not usable for individual decisions.

## Methodology Notes

- **File format:** the distributed file carried a `.csv` extension but was an
  XLSX workbook, verified via its ZIP/PK byte signature.
- **Duplicates:** 2 exact duplicate rows removed (397 -> 395).
- **Leakage control:** G1 and G2 are reported as separate scenarios rather than
  mixed into a single model. Including both raises regression R2 from 0.162 to
  0.724 without adding insight.
- **Metric selection:** tuning on `f1` optimises the majority class only and
  selected a degenerate model. `balanced_accuracy` was used instead.
- **Outliers:** the IQR rule flags 83 values in `failures` because its IQR is
  zero — invalid for ordinal scales. Conversely it flags none of the 38 students
  with G3 = 0, which the histogram shows as a clearly separate cluster.
- **Administrative artifact:** all 38 students with G3 = 0 also have 0 recorded
  absences, indicating deregistration rather than attendance. Removing the
  feature changed balanced accuracy by 0.000.

## Repository Structure
## Reproduction

```bash
pip install -r requirements.txt
jupyter notebook student_performance_analysis.ipynb
```

All results use `random_state=42` and an 80/20 stratified split.

## Limitations

- n = 395 from two Portuguese schools; not generalisable without revalidation.
- Cross-sectional data; no causal claims. `famsup` shows a negative coefficient,
  most plausibly reverse causation (families engage *because* of poor grades).
- Self-reported variables (study time, alcohol use, going out) carry response bias.

## Data Source

Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school
student performance. *Proceedings of 5th FUture BUsiness TEChnology Conference*.
UCI Machine Learning Repository.
