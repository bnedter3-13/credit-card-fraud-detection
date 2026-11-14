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

numeric_cols = ["precision", "recall", "f1_score", "f2_score", "accuracy", "f1_gap", "pr_auc", "precision_at_100", "recall_at_100", "threshold"]
for col in numeric_cols:
    if col in comparison_df.columns:
        comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce")

# Best model by F1 — same rule as the notebook export (only models with a continuous score are eligible)
best_row = comparison_df.dropna(subset=["pr_auc"]).sort_values("f1_score", ascending=False).iloc[0]
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
bins = np.linspace(predictions_df["Fraud_Probability"].min(), predictions_df["Fraud_Probability"].max(), 41)
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
st.dataframe(top20, width="stretch")

st.divider()

# ---- Model comparison ----
st.subheader(f"Model Comparison — All {len(comparison_df)} Combinations")

# Filters (everything below reads only the precomputed CSV — no retraining)
f1, f2, f3, f4 = st.columns(4)
models = sorted(comparison_df["model"].unique())
variants = ["original", "undersampling", "smote"]
sel_models = f1.multiselect("Model", models, default=models)
sel_variants = f2.multiselect("Variant", variants, default=variants)
label_filter = f3.selectbox("Training labels", ["All", "Label-free only", "Uses labels only"])
top_n = f4.slider("Rows in chart (top N by F1)", 5, max(len(comparison_df), 6), 20)

view = comparison_df[comparison_df["model"].isin(sel_models) & comparison_df["variant"].isin(sel_variants)]
if "uses_labels" in view.columns and label_filter != "All":
    want = label_filter == "Uses labels only"
    view = view[view["uses_labels"].astype(str).str.lower().eq(str(want).lower())]
view = view.sort_values("f1_score", ascending=False).reset_index(drop=True)

st.caption("`uses_labels` is True for supervised models, `kmeans` (cluster mapping via `y_train`), `autoencoder_v2` (Class=0 rows only) and anything fit on an undersampled/SMOTE variant.")
st.dataframe(view, width="stretch")
st.caption(
    "`f1_gap` = train F1 − test F1 (same decision rule on the data each model was fitted on). A large positive gap "
    "(e.g. Random Forest: train F1 1.0) suggests overfitting, or a train/test distribution shift for resampled variants; "
    "low F1 on both sides suggests underfitting. Unsupervised models never learn from labels, so their gap reflects score "
    "distributions, not classic overfitting. One random split with 98 test frauds — don't over-read small gaps."
)
st.caption(
    "**Accuracy is misleading on imbalanced data:** only ~0.17% of transactions are fraud, so a model that "
    "never flags anything scores ~99.83% accuracy while catching 0 frauds. Judge models by precision, recall, "
    "F1 / F2 (F2 weighs recall 4x as much as precision), PR-AUC and Precision@100 instead."
)

if view.empty:
    st.info("No rows match the current filters.")
else:
    chart_df = view.head(top_n).iloc[::-1].reset_index(drop=True)
    chart_df["combo"] = chart_df["model"] + " / " + chart_df["variant"]

    y = np.arange(len(chart_df))
    height = 0.2
    fig2, ax2 = plt.subplots(figsize=(12, 0.6 * len(chart_df) + 2))
    ax2.barh(y - 1.5 * height, chart_df["precision"], height, label="Precision", color="#4C72B0")
    ax2.barh(y - 0.5 * height, chart_df["recall"], height, label="Recall", color="#DD8452")
    ax2.barh(y + 0.5 * height, chart_df["f1_score"], height, label="F1-score", color="#55A868")
    ax2.barh(y + 1.5 * height, chart_df["pr_auc"], height, label="PR-AUC", color="#8172B2")
    ax2.set_yticks(y)
    ax2.set_yticklabels(chart_df["combo"], fontsize=9)
    ax2.set_xlabel("Score")
    ax2.set_xlim(0, 1.05)
    ax2.set_title(f"Top {len(chart_df)} combinations by F1 (filters applied)")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig2)

    # Precision@100 — works for every model with a continuous score
    if "precision_at_100" in view.columns:
        st.subheader("Precision@100 (of the 100 highest-scoring test transactions, how many are fraud)")
        p100 = view.dropna(subset=["precision_at_100"]).sort_values("precision_at_100", ascending=False)
        st.dataframe(
            p100[["model", "variant", "uses_labels", "precision_at_100", "recall_at_100", "pr_auc"]].reset_index(drop=True),
            width="stretch",
        )
