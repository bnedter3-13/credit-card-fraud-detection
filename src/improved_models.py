"""Improvement round: feature selection + novelty detection + tuned supervised models.

Ported from `fraud_detection_improved.ipynb`. It is a separate step that runs AFTER the main
notebook has exported `reports/model_comparison.csv`; it only APPENDS new rows to that file
(existing rows are never rewritten) and is idempotent (re-running replaces its own rows).

Protocol (no test-set tuning):
  * The training set is split again 80/20 (stratified, random_state=42) into fit / validation.
  * Features are ranked with ANOVA F (f_classif) on the FIT part only.
  * Detectors are fitted on NORMAL-only fit rows using the top-k features; k is chosen by
    validation PR-AUC. The test set is scored once, for reporting.
  * Score-based detectors flag the top 0.17% of test scores; supervised rows use thresholds
    chosen on the validation set (F1 or F2).

Every new row has uses_labels=True (feature ranking / normal-only training / supervised models use
y_train), except `isolation_forest_baseline`, which uses no labels at all.

Seeds: the original improvement notebook uses random_state=0 for the detectors (and one shared
RandomState(0) for the 40K subsamples) and 42 for Random Forest. They are kept as-is so the numbers
reproduce the notebook; this differs from the main notebook's random_state=42 convention.
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parent.parent
Q = 0.0017  # flag the top 0.17% most anomalous test rows (= real fraud rate); TEST_CONTAMINATION
KS = [6, 8, 10, 12, 15]
DETECTOR_SEED = 0  # as in the improvement notebook (see module docstring)
RF_SEED = 42

# Row names added by this step (used to keep the step idempotent)
NEW_MODELS = [
    "isolation_forest_baseline",
    "isolation_forest_fs_novelty",
    "gmm_fs_novelty",
    "lof_fs_novelty",
    "elliptic_envelope_fs_novelty",
    "unsup_rank_ensemble",
    "random_forest_balanced_f1",
    "random_forest_balanced_f2",
    "hybrid_random_forest_balanced_f1",
]


def _load(data_dir):
    Xtr = pd.read_csv(Path(data_dir) / "X_train.csv")
    Xte = pd.read_csv(Path(data_dir) / "X_test.csv")
    ytr = pd.read_csv(Path(data_dir) / "y_train.csv").iloc[:, -1].values
    yte = pd.read_csv(Path(data_dir) / "y_test.csv").iloc[:, -1].values
    return Xtr.values, Xte.values, ytr, yte


def _best_threshold(y, s, beta=1):
    """Threshold maximizing F-beta on the validation set (grid of validation-score quantiles)."""
    ts = np.unique(np.quantile(s, np.linspace(0.9, 0.99999, 400)))
    sc = [fbeta_score(y, s >= t, beta=beta, zero_division=0) for t in ts]
    return ts[int(np.argmax(sc))]


def _precision_recall_at_k(y_true, score, k=100):
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    tp_k = int(np.asarray(y_true)[top].sum())
    return tp_k / k, tp_k / int(np.asarray(y_true).sum())


def _row(name, uses_labels, y_true, score, thr=None, report_threshold=False):
    """One result row in the model_comparison.csv column layout."""
    t = thr if thr is not None else np.quantile(score, 1 - Q)
    pred = (score >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    p100, r100 = _precision_recall_at_k(y_true, score)
    return {
        "model": name,
        "variant": "original",  # fitted on the original (non-resampled) training split
        "uses_labels": bool(uses_labels),
        "threshold": round(float(thr), 4) if report_threshold else np.nan,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "f2_score": round(fbeta_score(y_true, pred, beta=2, zero_division=0), 4),
        "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
        "pr_auc": round(average_precision_score(y_true, score), 4),
        "precision_at_100": round(p100, 4),
        "recall_at_100": round(r100, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        # train_* / f1_gap are not computed for this round -> left empty
    }


def run_improved_rows(data_dir):
    """Fit everything and return (list of result dicts, dict of diagnostics)."""
    Xtr, Xte, ytr, yte = _load(data_dir)
    cols = pd.read_csv(Path(data_dir) / "X_train.csv", nrows=0).columns.tolist()
    print(f"train {Xtr.shape} frauds {ytr.sum()} | test {Xte.shape} frauds {yte.sum()}")

    # Re-split the TRAINING set; k and thresholds are chosen on validation only
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=42)
    print(f"fit {Xa.shape} ({ya.sum()} fraud) | validation {Xv.shape} ({yv.sum()} fraud)")
    Xa_norm = Xa[ya == 0]  # normal-only data for novelty detection
    F, _ = f_classif(Xa, ya)  # feature ranking from the FIT part only
    order = np.argsort(-F)
    rng = np.random.RandomState(DETECTOR_SEED)  # one shared stream, used in the same order as the notebook

    rows, diag = [], {"top_features": [cols[i] for i in order[:15]]}

    # Baseline: label-free Isolation Forest on all features (fit on the full training set)
    t0 = time.time()
    iso = IsolationForest(n_estimators=300, random_state=DETECTOR_SEED, n_jobs=-1).fit(Xtr)
    s_base = -iso.score_samples(Xte)
    rows.append(_row("isolation_forest_baseline", False, yte, s_base))
    print(f"isolation_forest_baseline: F1={rows[-1]['f1_score']} ({time.time() - t0:.0f}s)")

    def sub(A):
        return A[rng.choice(len(A), 40000, replace=False)]

    makers = {
        "isolation_forest": lambda A: IsolationForest(
            n_estimators=300, random_state=DETECTOR_SEED, n_jobs=-1).fit(A),
        "gmm": lambda A: GaussianMixture(n_components=3, random_state=DETECTOR_SEED).fit(A),
        "lof": lambda A: LocalOutlierFactor(n_neighbors=30, novelty=True, n_jobs=-1).fit(sub(A)),
        "elliptic_envelope": lambda A: EllipticEnvelope(
            support_fraction=0.99, random_state=DETECTOR_SEED).fit(sub(A)),
    }
    test_scores, best_k, val_pr_auc = {}, {}, {}
    for name, make in makers.items():
        t0 = time.time()
        best = (-1.0, None, None)
        val_pr_auc[name] = {}
        for k in KS:
            idx = order[:k]
            model = make(Xa_norm[:, idx])
            apv = average_precision_score(yv, -model.score_samples(Xv[:, idx]))  # validation PR-AUC
            val_pr_auc[name][k] = round(apv, 3)
            if apv > best[0]:
                best = (apv, k, model)
        best_k[name] = best[1]
        test_scores[name] = -best[2].score_samples(Xte[:, order[: best[1]]])
        rows.append(_row(f"{name}_fs_novelty", True, yte, test_scores[name]))
        print(f"{name}_fs_novelty (k={best[1]}): F1={rows[-1]['f1_score']} ({time.time() - t0:.0f}s)")
    diag["best_k"], diag["val_pr_auc_by_k"] = best_k, val_pr_auc

    # Rank-average ensemble of GMM + LOF + Elliptic Envelope
    rank = lambda s: rankdata(s) / len(s)
    ens = np.mean([rank(test_scores[n]) for n in ["gmm", "lof", "elliptic_envelope"]], axis=0)
    rows.append(_row("unsup_rank_ensemble", True, yte, ens))
    print(f"unsup_rank_ensemble: F1={rows[-1]['f1_score']}")

    # Supervised: class-balanced Random Forest with validation-tuned thresholds
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=RF_SEED).fit(Xa, ya)
    sv, st = rf.predict_proba(Xv)[:, 1], rf.predict_proba(Xte)[:, 1]
    thr1, thr2 = _best_threshold(yv, sv, 1), _best_threshold(yv, sv, 2)
    rows.append(_row("random_forest_balanced_f1", True, yte, st, thr=thr1, report_threshold=True))
    rows.append(_row("random_forest_balanced_f2", True, yte, st, thr=thr2, report_threshold=True))

    # Hybrid: add the label-free Isolation Forest score (fitted on the fit part) as a feature
    iso_a = IsolationForest(n_estimators=300, random_state=DETECTOR_SEED, n_jobs=-1).fit(Xa)
    add = lambda X: np.c_[X, -iso_a.score_samples(X)]
    rfh = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RF_SEED).fit(add(Xa), ya)
    svh, sth = rfh.predict_proba(add(Xv))[:, 1], rfh.predict_proba(add(Xte))[:, 1]
    rows.append(_row("hybrid_random_forest_balanced_f1", True, yte, sth,
                     thr=_best_threshold(yv, svh, 1), report_threshold=True))
    print(f"random forests: F1={[r['f1_score'] for r in rows[-3:]]} ({time.time() - t0:.0f}s)")
    return rows, diag


def append_rows(csv_path, rows):
    """Append rows to model_comparison.csv without touching existing lines.

    Idempotent: lines written by an earlier run of this step (model names in NEW_MODELS) are dropped
    first; every other line is kept byte-for-byte.
    """
    csv_path = Path(csv_path)
    with open(csv_path, newline="") as fh:
        lines = fh.read().splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    kept = [lines[0]] + [ln for ln in lines[1:] if ln.split(",", 1)[0] not in NEW_MODELS]
    if not kept[-1].endswith("\n"):
        kept[-1] += "\n"
    new_lines = []
    for r in rows:
        assert set(r) <= set(header), set(r) - set(header)
        vals = []
        for col in header:
            v = r.get(col, np.nan)
            vals.append("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v))
        new_lines.append(",".join(vals) + "\n")
    with open(csv_path, "w", newline="") as fh:
        fh.writelines(kept + new_lines)
    return len(new_lines)


def run_improved_round(data_dir=ROOT / "data" / "processed",
                       comparison_csv=ROOT / "reports" / "model_comparison.csv"):
    start = time.time()
    rows, diag = run_improved_rows(data_dir)
    n = append_rows(comparison_csv, rows)
    print(f"Appended {n} rows to {comparison_csv} ({time.time() - start:.0f}s total)")
    return pd.DataFrame(rows), diag


if __name__ == "__main__":
    run_improved_round()
