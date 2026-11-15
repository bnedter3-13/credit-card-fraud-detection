# Credit Card Fraud Detection

Comparing unsupervised anomaly detection against a supervised benchmark for catching credit card fraud.

## Overview

This project detects fraudulent credit card transactions using eight unsupervised anomaly detectors (Isolation Forest, Local Outlier Factor, K-Means, One-Class SVM, DBSCAN, Gaussian Mixture, Elliptic Envelope, MLP autoencoder), benchmarked against two supervised classifiers (Random Forest, Logistic Regression). Follow-up experiments add validation-tuned thresholds (`random_forest_tuned`, `logistic_regression_tuned`), a normal-rows-only `autoencoder_v2`, a label-free `kmeans_distance`, and a hybrid Random Forest using an Isolation Forest score as an extra feature (`hybrid_random_forest`, `hybrid_random_forest_tuned`). Each of the 10 base models is trained on three class-imbalance strategies — original, undersampling, and SMOTE — to see how resampling affects unsupervised vs. supervised methods. The data is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud): 284,807 transactions, only ~0.17% of which are fraudulent.

## Key Results

All numbers come from [`reports/model_comparison.csv`](reports/model_comparison.csv): 76 rows — 63 from the main notebook (10 algorithms plus the extra experiments, each on original / undersampling / SMOTE training data), 9 from the improvement round and 4 from the Elliptic Envelope follow-up below — all scored on the same untouched test set (56,962 transactions, 98 fraud). The headline results below are from the main 63 rows.

The best row by test F1 is **`hybrid_random_forest_tuned` trained on the original (unresampled) data**: a Random Forest with an Isolation Forest anomaly score as an extra feature and a validation-chosen threshold (0.44).

| Metric    | Score |
|-----------|-------|
| F1-score  | 0.880 |
| Precision | 94.2% |
| Recall    | 82.7% |
| PR-AUC    | 0.867 |

It beats plain Random Forest / original (F1 0.868) by only 2 caught frauds out of 98 on one test split, so the top four rows (hybrid tuned 0.880, Random Forest 0.868, hybrid 0.867, Random Forest tuned 0.863) are close to tied.

Best row per base algorithm:

| Model | Variant | Precision | Recall | F1 | PR-AUC |
|-------|---------|-----------|--------|----|--------|
| Random Forest (supervised) | original | 0.941 | 0.806 | 0.868 | 0.861 |
| Logistic Regression (supervised) | original | 0.829 | 0.643 | 0.724 | 0.744 |
| K-Means (cluster labelled via `y_train`) | undersampling / smote | 0.750 | 0.337 | 0.465 | N/A |
| LOF | smote | 0.233 | 0.520 | 0.322 | 0.181 |
| Isolation Forest | original | 0.308 | 0.327 | 0.317 | 0.192 |
| DBSCAN | original | 0.177 | 0.612 | 0.275 | 0.214 |
| Gaussian Mixture | original | 0.247 | 0.245 | 0.246 | 0.155 |
| One-Class SVM | original | 0.058 | 0.806 | 0.108 | 0.137 |
| Autoencoder (MLPRegressor) | original | 0.062 | 0.061 | 0.062 | 0.047 |
| Elliptic Envelope | any | 0.000 | 0.000 | 0.000 | ≤0.029 |

What else the experiments showed:

- **Label-free models:** by Precision@100 (how many of the 100 highest-scoring test transactions are fraud), Isolation Forest / original leads at 31%, then DBSCAN 26%, GMM 24%, `kmeans_distance` 24% and One-Class SVM 22%. `autoencoder_v2` (trained on `Class=0` rows, so it uses labels) reaches F1 0.400 and Precision@100 of 41%, against 0.062 and 6% for the first autoencoder; its undersampling fit did not converge.
- **Threshold tuning** lifted resampled supervised models (Random Forest / undersampling F1 0.081 to 0.747) but not Random Forest / original (0.868 to 0.863).
- **Recall vs. precision:** `random_forest_f2` / original (threshold tuned for F2) raised recall to 87.8% (vs. 80.6%) with precision 82.7% (vs. 94.1%) and F1 0.852 (vs. 0.868). Across three model seeds its recall was 0.874 ± 0.006 but its F1 varied 0.831 ± 0.046, so the F1 change is within noise. Rows with higher recall, such as `logistic_regression_cw` (90.8%), have precision of only about 6%. The Isolation Forest ensembles were worse than Random Forest alone.
- **Train vs. test:** Random Forest / original has train F1 1.000 vs. test 0.868 (gap +0.13, typical tree-ensemble overfitting); Logistic Regression's gap is +0.02; the unsupervised models have near-zero or slightly negative gaps, and LOF and Elliptic Envelope score 0 on both sides.
- **Accuracy is not useful here:** across the 63 main rows it runs from 0.473 to 0.9996, the 28 rows with F1 ≥ 0.5 all sit at 0.998 or above, and models that catch no fraud still score 0.996–0.998.
- One-Class SVM (30k rows), DBSCAN (20k rows) and Elliptic Envelope (30k rows) are fit on subsamples.

Details for every row (F2, accuracy, Precision@100, `uses_labels`, `train_f1`, `f1_gap`) are in the CSV and in Section 8 of the notebook.

### Elliptic Envelope follow-up

A label-free Elliptic Envelope (fitted on the 80% train part, no class filtering) is very sensitive to `support_fraction`. Choosing it by validation PR-AUC from {None, 0.75, 0.9, 0.95, 0.99} picked **0.99** (validation PR-AUC 0.679; the smaller values scored 0.60 or less and the default `None` scored 0.009).

| Row (`uses_labels`) | Rule | Precision | Recall | F1 | PR-AUC | Precision@100 |
|---------------------|------|-----------|--------|----|--------|---------------|
| `support_fraction=0.99` (True) | `predict` | 0.734 | 0.816 | 0.773 | 0.684 | 0.80 |
| `support_fraction=0.99` (True) | top 0.17% | 0.825 | 0.816 | 0.820 | 0.684 | 0.80 |
| default `None` (False) | `predict` | 0.000 | 0.000 | 0.000 | 0.010 | 0.00 |
| default `None` (False) | top 0.17% | 0.000 | 0.000 | 0.000 | 0.010 | 0.00 |

- **Assumption:** the top-0.17% rule uses the known fraud rate (it flags 0.17% of test rows), and so does `predict` through `contamination=0.0017`. A real system would not know this rate in advance.
- **Validation labels:** `support_fraction` was chosen using validation labels, so the chosen rows are marked `uses_labels=True`; only the untuned default is label-free (and scores 0).
- **Seeds:** the chosen fit uses all training-part rows and is deterministic (identical F1 over 3 seeds). Forced onto a 40k subsample, F1 is 0.759 ± 0.055 for `predict` and 0.8205 ± 0.000 for the top-0.17% rule.
- **Versus Random Forest** (1,000 bootstrap resamples, 95% CI for F1): Elliptic Envelope 0.773 [0.705, 0.833] and 0.820 [0.757, 0.874]; Random Forest 0.868 [0.811, 0.918]; hybrid tuned 0.880 [0.827, 0.926]. The paired F1 difference to Random Forest is −0.096 and −0.048 for the Elliptic Envelope (intervals exclude 0) and +0.012 [0.000, 0.030] for the hybrid. McNemar on all rows: Elliptic Envelope vs. Random Forest p = 3.4e-05 (`predict`) and 0.019 (top 0.17%), hybrid vs. Random Forest p = 0.50; on the 98 fraud rows alone there is no difference (p = 1.0 / 0.50). The Elliptic Envelope finds as many frauds as Random Forest (80 vs. 79) but with more false positives (17–29 vs. 5).
- **Preprocessing change:** `Amount` and `Time` are now standardized with scalers fit on the training split only (before, on all rows). 63 of the 72 earlier rows are unchanged; the largest F1 change is 0.016 (One-Class SVM / SMOTE, 0.057 → 0.041), and the headline rows are unchanged.

## Improvement Round (semi-supervised)

A separate step, [`src/improved_models.py`](src/improved_models.py), runs after the main notebook export and appends 9 rows to the CSV. The training set is split again 80/20 into fit / validation; features are ranked with ANOVA F on the fit part, detectors are fitted on normal-only rows with the top-k features (k chosen by validation PR-AUC, never on the test set), and the top 0.17% of test scores are flagged. Supervised rows use thresholds tuned on the validation set.

**These methods use labels** (feature ranking and normal-only training use `y_train`), so they are semi-supervised, not unsupervised; only the baseline row is label-free.

| Model | Precision | Recall | F1 | F2 | Accuracy | PR-AUC |
|-------|-----------|--------|----|----|----------|--------|
| `random_forest_balanced_f1` (F1-tuned threshold) | 0.901 | 0.837 | 0.868 | 0.849 | 99.96% | 0.862 |
| `hybrid_random_forest_balanced_f1` (+ Isolation Forest score) | 0.863 | 0.837 | 0.850 | 0.842 | 99.95% | 0.856 |
| `unsup_rank_ensemble` (GMM + LOF + Elliptic Envelope) | 0.806 | 0.806 | 0.806 | 0.806 | 99.93% | 0.714 |
| `lof_fs_novelty` (k=6) | 0.794 | 0.786 | 0.790 | 0.787 | 99.93% | 0.764 |
| `elliptic_envelope_fs_novelty` (k=15) | 0.794 | 0.786 | 0.790 | 0.787 | 99.93% | 0.684 |
| `random_forest_balanced_f2` (F2-tuned, recall-oriented) | 0.642 | 0.898 | 0.749 | 0.832 | 99.90% | 0.862 |
| `gmm_fs_novelty` (k=6) | 0.732 | 0.724 | 0.728 | 0.726 | 99.91% | 0.668 |
| `isolation_forest_fs_novelty` (k=10) | 0.670 | 0.663 | 0.667 | 0.665 | 99.89% | 0.616 |
| `isolation_forest_baseline` (label-free, all features) | 0.299 | 0.296 | 0.297 | 0.296 | 99.76% | 0.190 |

- Feature selection plus normal-only training lifts the detectors from F1 0.30 (label-free Isolation Forest) to 0.67–0.81, but only because they use labels.
- Accuracy is 99.76–99.96% for every row here and says almost nothing: flagging nothing already scores 99.83%.
- The test set has only 98 frauds, so one extra caught fraud moves recall by about 1 point. Differences of a few points between these rows are within noise; the improved Random Forest (F1 0.868) is not distinguishable from the main notebook's best Random Forest rows (0.863–0.880).
- The detectors keep the `random_state=0` of the original improvement notebook (Random Forest uses 42), which is why its numbers reproduce within ±0.01.
- Run it alone with `python -m src.improved_models` (needs `reports/model_comparison.csv` from the main notebook).

## Project Structure

```
notebooks/        Main analysis notebook — EDA, preprocessing, modeling, evaluation
src/              Improvement-round step (improved_models.py), called from the notebook
dashboard/        Streamlit app for exploring results (app.py)
data/raw/         Place creditcard.csv here (not committed)
data/processed/   Train/test splits written by the notebook (not committed)
reports/          CSVs exported by the notebook and read by the dashboard
requirements.txt  Pinned dependencies
```

## How to Run

Tested with Python 3.11 and 3.12.

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Get the dataset**
   Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection page](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in `data/raw/`.

3. **Run the notebook** (takes about 15 minutes)
   ```
   jupyter notebook notebooks/fraud_detection.ipynb
   ```
   Run all cells top to bottom — this reproduces the full analysis, exports the CSVs the dashboard needs into `reports/`, and runs the improvement round (Section 8e).

4. **Launch the dashboard**
   ```
   streamlit run dashboard/app.py
   ```

## Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · SciPy · matplotlib / seaborn · Streamlit · Jupyter
