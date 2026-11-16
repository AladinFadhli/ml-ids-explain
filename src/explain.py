"""Tasks 2-6: global importances, local SHAP alerts, ablation, stability."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from . import config as C


# ---------- SHAP --------------------------------------------------------------
def shap_explainer(model):
    return shap.TreeExplainer(model)


def shap_values_pos(explainer, X):
    """SHAP values for the 'attack' class as a 2-D array (rows x features), plus the base value."""
    sv = explainer.shap_values(X, check_additivity=False)
    base = explainer.expected_value
    if isinstance(sv, list):                 # old shap: one array per class
        sv, base = sv[1], base[1]
    elif sv.ndim == 3:                       # new shap: (rows, features, classes)
        sv, base = sv[:, :, 1], np.atleast_1d(base)[-1]
    return sv, float(np.atleast_1d(base)[-1])


# ---------- task 2: three global measures --------------------------------------
def stratified_rows(y, n, seed=C.SEED):
    if len(y) <= n:
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def global_importances(rf, X_test, y_test, sv_abs_mean, feat, seed=C.SEED):
    rows = stratified_rows(y_test, C.PERM_MAX_ROWS, seed)
    perm = permutation_importance(rf, X_test.iloc[rows], y_test.iloc[rows], scoring="average_precision",
                                  n_repeats=C.PERM_REPEATS, random_state=seed, n_jobs=1)
    df = pd.DataFrame({
        "impurity": rf.feature_importances_,
        "permutation_test": perm.importances_mean,
        "permutation_std": perm.importances_std,
        "mean_abs_shap": sv_abs_mean,
    }, index=feat)
    for m in ["impurity", "permutation_test", "mean_abs_shap"]:
        df[f"rank_{m}"] = df[m].rank(ascending=False, method="min").astype(int)
    return df.sort_values("mean_abs_shap", ascending=False)


def rank_agreement(imp: pd.DataFrame, k=C.TOP_K) -> pd.DataFrame:
    ms = ["impurity", "permutation_test", "mean_abs_shap"]
    out = []
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            ta = set(imp[a].nlargest(k).index)
            tb = set(imp[b].nlargest(k).index)
            tau = kendalltau(imp[a], imp[b]).statistic
            out.append({"pair": f"{a} vs {b}", f"shared top-{k}": len(ta & tb), "kendall_tau_all": round(tau, 3)})
    return pd.DataFrame(out)


def correlated_pairs(X, feats, thr=0.95):
    corr = X[feats].corr().abs()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if corr.loc[a, b] >= thr]
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_corr"]).sort_values("abs_corr", ascending=False)


def plot_importances(imp: pd.DataFrame, k=C.TOP_K, path=None):
    top = imp.head(k).iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * k + 1.5), sharey=True)
    for ax, m, t in zip(axes, ["impurity", "permutation_test", "mean_abs_shap"],
                        ["RF impurity (train)", "Permutation on test (PR-AUC drop)", "mean |SHAP| (test sample)"]):
        ax.barh(top.index, top[m])
        ax.set_title(t, fontsize=10)
    fig.suptitle(f"Top {k} features, ordered by mean |SHAP|")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# ---------- task 3: five local explanations -------------------------------------
def pick_alerts(y, p, thr):
    """2 TP (most confident, and closest to the threshold), 2 FP (most confident), 1 FN (most confident miss)."""
    y, p = np.asarray(y), np.asarray(p)
    tp = np.where((y == 1) & (p >= thr))[0]
    fp = np.where((y == 0) & (p >= thr))[0]
    fn = np.where((y == 1) & (p < thr))[0]
    chosen = []
    if len(tp):
        chosen.append(("TP - confident", tp[np.argmax(p[tp])]))
        chosen.append(("TP - borderline", tp[np.argmin(p[tp])]))
    for i, j in enumerate(fp[np.argsort(-p[fp])][:2]):
        chosen.append((f"FP #{i + 1}", j))
    if len(fn):
        chosen.append(("FN - missed attack", fn[np.argmin(p[fn])]))
    return chosen


def shap_sample_rows(y, alert_rows, n=C.SHAP_MAX_ROWS, seed=C.SEED):
    """At most n test rows: the 5 alerts + a stratified sample."""
    base = stratified_rows(y, n - len(alert_rows), seed)
    return np.unique(np.concatenate([np.asarray(alert_rows, dtype=int), base]))[:n]


def alert_table(x_row, sv_row, feat, k=8):
    return (pd.DataFrame({"feature": feat, "value": np.asarray(x_row), "shap": sv_row})
            .assign(abs=lambda d: d.shap.abs()).nlargest(k, "abs").drop(columns="abs"))


def analyst_note_draft(kind, family, prob, thr, table):
    push = table[table.shap > 0].head(3)
    pull = table[table.shap < 0].head(2)
    up = "; ".join(f"{r.feature} = {r.value:g}" for r in push.itertuples()) or "none"
    down = "; ".join(f"{r.feature} = {r.value:g}" for r in pull.itertuples()) or "none"
    return (f"[{kind}] true class: {family} | score {prob:.3f} (threshold {thr:.3f})\n"
            f"  pushed towards ATTACK: {up}\n"
            f"  pushed towards BENIGN: {down}\n"
            f"  -> analyst: is this behaviour or a capture artefact? verdict + next step: ...")


def plot_waterfall(sv_row, base, x_row, feat, title, path=None):
    exp = shap.Explanation(values=sv_row, base_values=base, data=np.asarray(x_row), feature_names=feat)
    plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------- task 6: stability --------------------------------------------------
def stability_run(rf_template, train, test, feat, seed, n_shap=1_000):
    rf = clone(rf_template).set_params(random_state=seed)
    rf.fit(train[feat], train["y"])
    rows = stratified_rows(test["y"], n_shap, seed)
    sv, _ = shap_values_pos(shap_explainer(rf), test[feat].iloc[rows])
    return pd.Series(np.abs(sv).mean(axis=0), index=feat)


def stability_table(runs: dict, k=10) -> pd.DataFrame:
    ref_name = next(iter(runs))
    ref = runs[ref_name]
    ref_top = set(ref.nlargest(k).index)
    out = []
    for name, s in runs.items():
        out.append({"run": name, f"shared top-{k} with {ref_name}": len(ref_top & set(s.nlargest(k).index)),
                    "kendall_tau_all": round(kendalltau(ref, s).statistic, 3),
                    "top-5": ", ".join(s.nlargest(5).index)})
    return pd.DataFrame(out)
