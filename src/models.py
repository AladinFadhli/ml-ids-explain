"""Baseline, Random Forest and XGBoost, tuned by stratified CV on the training set only."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C


def make_models(y_train, seed=C.SEED):
    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos
    return {
        "LogReg (baseline)": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=seed),
            {"max_depth": [20, None], "min_samples_leaf": [1, 5]},
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="aucpr",
                          scale_pos_weight=neg / pos, n_jobs=-1, random_state=seed),
            {"max_depth": [6, 10], "learning_rate": [0.1, 0.3]},
        ),
    }


def tuning_subset(X, y, family, seed=C.SEED):
    """Stratified subset of TRAIN for the CV grid search (CPU budget)."""
    if len(X) <= C.TUNE_MAX_ROWS:
        return X, y
    strat = family.where(family.map(family.value_counts()) >= 2, "rare")
    idx, _ = train_test_split(np.arange(len(X)), train_size=C.TUNE_MAX_ROWS, stratify=strat, random_state=seed)
    return X.iloc[idx], y.iloc[idx]


def tune_and_fit(name, estimator, grid, X_train, y_train, fam_train, seed=C.SEED):
    """Grid search on a train subset (PR-AUC), then refit the best params on the full train set."""
    Xs, ys = tuning_subset(X_train, y_train, fam_train, seed)
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=seed)
    gs = GridSearchCV(estimator, grid, scoring="average_precision", cv=cv, n_jobs=1, refit=False)
    t0 = time.time()
    gs.fit(Xs, ys)
    tune_s = time.time() - t0
    best = estimator.set_params(**gs.best_params_)
    t0 = time.time()
    best.fit(X_train, y_train)
    fit_s = time.time() - t0
    info = {"model": name, "best_params": gs.best_params_, "cv_pr_auc": gs.best_score_,
            "tune_s": tune_s, "fit_s": fit_s}
    return best, info


def predict_time(model, X):
    t0 = time.time()
    p = model.predict_proba(X)[:, 1]
    return p, (time.time() - t0) / len(X) * 1e6   # microseconds per flow
