"""Metrics for imbalanced detection: confusion, P/R/F1, PR-AUC, alert load. Never accuracy alone."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def best_threshold(y_val, p_val):
    """Threshold that maximises F1 on VALIDATION. The test set never chooses anything."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def scores(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": thr,
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision_benign": tn / max(tn + fn, 1),
        "recall_benign": tn / max(tn + fp, 1),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "alerts_per_10k_flows": (tp + fp) / len(y) * 10_000,
        "false_alerts_per_10k_flows": fp / len(y) * 10_000,
        "missed_attacks": int(fn),
    }


def plot_pr(curves: dict, base_rate: float, title: str, path=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(base_rate, ls="--", c="grey", lw=1, label=f"random = attack rate {base_rate:.3f}")
    ax.set_xlabel("Recall (attacks caught)")
    ax.set_ylabel("Precision (alerts that are real)")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_confusion(y, p, thr, title, path=None):
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["benign", "attack"])
    ax.set_yticks([0, 1], ["benign", "attack"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def recall_by_family(family: pd.Series, y, p, thr) -> pd.DataFrame:
    df = pd.DataFrame({"family": family.values, "y": np.asarray(y), "hit": (np.asarray(p) >= thr)})
    att = df[df.y == 1]
    out = att.groupby("family").agg(flows=("hit", "size"), detected=("hit", "sum"))
    out["recall"] = out["detected"] / out["flows"]
    ben = df[df.y == 0]
    out.loc["BENIGN (false alert rate)"] = [len(ben), int(ben.hit.sum()), ben.hit.mean()]
    return out.sort_values("flows", ascending=False)
