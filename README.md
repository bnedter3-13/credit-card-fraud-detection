# Credit Card Fraud Detection

Comparing unsupervised anomaly detection against a supervised benchmark for catching credit card fraud.

## Overview

This project detects fraudulent credit card transactions using unsupervised anomaly detection (Isolation Forest, Local Outlier Factor, K-Means clustering), benchmarked against a supervised Random Forest classifier. Each model is trained on three class-imbalance strategies — original, undersampling, and SMOTE — to see how resampling affects unsupervised vs. supervised methods. The data is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud): 284,807 transactions, only ~0.17% of which are fraudulent.

## Key Results

The best combination overall was **Random Forest trained on the original (unresampled) data**:

| Metric    | Score |
|-----------|-------|
| F1-score  | 0.868 |
| Precision | 94.1% |
| Recall    | 80.6% |
| PR-AUC    | 0.86  |

Full results for all 12 model × resampling-strategy combinations are in [`reports/model_comparison.csv`](reports/model_comparison.csv) and Section 8 of the notebook.

## Project Structure

```
notebooks/    Main analysis notebook — EDA, preprocessing, modeling, evaluation
dashboard/    Streamlit app for exploring results (app.py)
data/         Raw and processed datasets (not committed — see "How to Run")
reports/      Exported CSVs (model comparison, best model's predictions) used by the dashboard
```

## How to Run

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Get the dataset**
   Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection page](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in `data/raw/`.

3. **Run the notebook**
   ```
   jupyter notebook notebooks/fraud_detection.ipynb
   ```
   Run all cells top to bottom — this reproduces the full analysis and exports the CSVs the dashboard needs into `reports/`.

4. **Launch the dashboard**
   ```
   streamlit run dashboard/app.py
   ```

## Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · matplotlib / seaborn · Streamlit · Jupyter
