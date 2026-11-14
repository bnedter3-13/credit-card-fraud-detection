# Credit Card Fraud Detection

Comparing unsupervised anomaly detection against a supervised benchmark for catching credit card fraud.

## Overview

This project detects fraudulent credit card transactions using eight unsupervised anomaly detectors (Isolation Forest, Local Outlier Factor, K-Means, One-Class SVM, DBSCAN, Gaussian Mixture, Elliptic Envelope, MLP autoencoder), benchmarked against two supervised classifiers (Random Forest, Logistic Regression). Follow-up experiments add validation-tuned thresholds (`random_forest_tuned`, `logistic_regression_tuned`), a normal-rows-only `autoencoder_v2`, a label-free `kmeans_distance`, and a hybrid Random Forest using an Isolation Forest score as an extra feature (`hybrid_random_forest`, `hybrid_random_forest_tuned`). Each of the 10 base models is trained on three class-imbalance strategies — original, undersampling, and SMOTE — to see how resampling affects unsupervised vs. supervised methods. The data is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud): 284,807 transactions, only ~0.17% of which are fraudulent.

## Key Results

All numbers come from [`reports/model_comparison.csv`](reports/model_comparison.csv): 63 rows (10 algorithms plus the extra experiments, each on original / undersampling / SMOTE training data), all scored on the same untouched test set (56,962 transactions, 98 fraud).

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
| One-Class SVM | original | 0.058 | 0.796 | 0.108 | 0.137 |
| Autoencoder (MLPRegressor) | original | 0.062 | 0.061 | 0.062 | 0.047 |
| Elliptic Envelope | any | 0.000 | 0.000 | 0.000 | ≤0.029 |

What else the experiments showed:

- **Label-free models:** by Precision@100 (how many of the 100 highest-scoring test transactions are fraud), Isolation Forest / original leads at 31%, then DBSCAN 26%, GMM 24%, `kmeans_distance` 24% and One-Class SVM 22%. `autoencoder_v2` (trained on `Class=0` rows, so it uses labels) reaches F1 0.400 and Precision@100 of 41%, against 0.062 and 6% for the first autoencoder; its undersampling fit did not converge.
- **Threshold tuning** lifted resampled supervised models (Random Forest / undersampling F1 0.081 to 0.747) but not Random Forest / original (0.868 to 0.863).
- **Recall vs. precision:** `random_forest_f2` / original (threshold tuned for F2) raised recall to 87.8% (vs. 80.6%) with precision 82.7% (vs. 94.1%) and F1 0.852 (vs. 0.868). Across three model seeds its recall was 0.874 ± 0.006 but its F1 varied 0.831 ± 0.046, so the F1 change is within noise. Rows with higher recall, such as `logistic_regression_cw` (90.8%), have precision of only about 6%. The Isolation Forest ensembles were worse than Random Forest alone.
- **Train vs. test:** Random Forest / original has train F1 1.000 vs. test 0.868 (gap +0.13, typical tree-ensemble overfitting); Logistic Regression's gap is +0.02; the unsupervised models have near-zero or slightly negative gaps, and LOF and Elliptic Envelope score 0 on both sides.
- **Accuracy is not useful here:** it runs from 0.473 to 0.9996, the 28 rows with F1 ≥ 0.5 all sit at 0.998 or above, and models that catch no fraud still score 0.996–0.998.
- One-Class SVM (30k rows), DBSCAN (20k rows) and Elliptic Envelope (30k rows) are fit on subsamples.

Details for every row (F2, accuracy, Precision@100, `uses_labels`, `train_f1`, `f1_gap`) are in the CSV and in Section 8 of the notebook.

## Project Structure

```
notebooks/        Main analysis notebook — EDA, preprocessing, modeling, evaluation
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

3. **Run the notebook** (takes about 14 minutes)
   ```
   jupyter notebook notebooks/fraud_detection.ipynb
   ```
   Run all cells top to bottom — this reproduces the full analysis and exports the CSVs the dashboard needs into `reports/`.

4. **Launch the dashboard**
   ```
   streamlit run dashboard/app.py
   ```

## Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · SciPy · matplotlib / seaborn · Streamlit · Jupyter
