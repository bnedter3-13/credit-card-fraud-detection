import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache_data
def load_data():
    comparison_df = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    predictions_df = pd.read_csv(REPORTS_DIR / "best_model_predictions.csv")
    return comparison_df, predictions_df


comparison_df, predictions_df = load_data()

st.title("Credit Card Fraud Detection Dashboard")
st.caption(
    "Results are precomputed in the analysis notebook — this dashboard only "
    "loads and displays `reports/model_comparison.csv` and "
    "`reports/best_model_predictions.csv`."
)

best_row = comparison_df.sort_values("f1_score", ascending=False).iloc[0]
total_transactions = len(predictions_df)
fraud_detected = int((predictions_df["Predicted_Class"] == 1).sum())
actual_fraud_rate = predictions_df["Class"].mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Fraud Detected (Best Model)", f"{fraud_detected:,}")
col3.metric("Actual Fraud Rate", f"{actual_fraud_rate:.3f}%")
col4.metric(
    "Best Model F1-score",
    f"{best_row['f1_score']:.4f}",
    help=f"{best_row['model']} / {best_row['variant']}",
)

st.divider()

st.subheader("Anomaly Score Distribution")
st.caption(
    f"`Fraud_Probability` from the best model ({best_row['model']} / "
    f"{best_row['variant']}), split by the true `Class` label."
)

fig1, ax1 = plt.subplots(figsize=(10, 4))
bins = np.linspace(0, 1, 41)
ax1.hist(
    predictions_df.loc[predictions_df["Class"] == 0, "Fraud_Probability"],
    bins=bins,
    alpha=0.6,
    label="Non-fraud",
    color="#4C72B0",
)
ax1.hist(
    predictions_df.loc[predictions_df["Class"] == 1, "Fraud_Probability"],
    bins=bins,
    alpha=0.6,
    label="Fraud",
    color="#C44E52",
)
ax1.set_yscale("log")
ax1.set_xlabel("Fraud Probability")
ax1.set_ylabel("Count (log scale)")
ax1.legend()
st.pyplot(fig1)

st.divider()

# ---- Top suspicious transactions ----
st.subheader("Top 20 Suspicious Transactions")
top20 = (
    predictions_df.sort_values("Fraud_Probability", ascending=False)
    .head(20)[["Time", "Amount", "Predicted_Class", "Fraud_Probability"]]
    .reset_index(drop=True)
)
st.dataframe(top20, use_container_width=True)

st.divider()

# ---- Model comparison ----
st.subheader("Model Comparison — All 12 Combinations")
st.dataframe(
    comparison_df.sort_values("f1_score", ascending=False).reset_index(drop=True),
    use_container_width=True,
)

chart_df = comparison_df.copy()
chart_df["combo"] = chart_df["model"] + " / " + chart_df["variant"]
chart_df["pr_auc_numeric"] = pd.to_numeric(chart_df["pr_auc"], errors="coerce")

x = np.arange(len(chart_df))
width = 0.2

fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.bar(x - 1.5 * width, chart_df["precision"], width, label="Precision", color="#4C72B0")
ax2.bar(x - 0.5 * width, chart_df["recall"], width, label="Recall", color="#DD8452")
ax2.bar(x + 0.5 * width, chart_df["f1_score"], width, label="F1-score", color="#55A868")
ax2.bar(x + 1.5 * width, chart_df["pr_auc_numeric"], width, label="PR-AUC", color="#8172B2")
ax2.set_xticks(x)
ax2.set_xticklabels(chart_df["combo"], rotation=45, ha="right")
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.05)
ax2.set_title("Precision / Recall / F1 / PR-AUC Across All Model × Variant Combinations")
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)
