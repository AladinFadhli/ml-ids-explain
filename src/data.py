"""Load, clean, sample and split CIC-IDS2017 without leaking the test set."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def normalise(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return name.strip("_")


def clean_family(label: str) -> str:
    """'Web Attack \x96 Brute Force' -> 'Web Attack - Brute Force' (the original CSVs carry a bad byte)."""
    s = re.sub(r"\s*[^\x20-\x7E]+\s*", " - ", str(label)).strip()
    return re.sub(r"\s+", " ", s)


def load_raw(data_dir: Path = C.DATA_DIR) -> pd.DataFrame:
    files = sorted(Path(data_dir).rglob("*.csv"))   # searches subfolders too (Kaggle)
    if not files:
        raise FileNotFoundError(f"No CSV in {data_dir}. See data/README.md.")
    parts = []
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = [normalise(c) for c in df.columns]
        df["source_file"] = f.name
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df


def clean(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Non-learned cleaning on the full table, before any split. Every step is logged."""
    n0 = len(df)
    log.append(("raw rows", n0))

    df = df.dropna(subset=[C.LABEL_COL])
    df["family"] = df[C.LABEL_COL].map(clean_family)
    df["family"] = df["family"].replace({"BENIGN": C.BENIGN, "Benign": C.BENIGN})

    attempted = df["family"].str.contains("attempted", case=False)
    log.append(("'Attempted' flows removed (corrected release only)", int(attempted.sum())))
    df = df[~attempted]

    drop_ids = [c for c in C.ID_COLUMNS if c in df.columns]
    log.append(("identifier columns dropped", ", ".join(drop_ids) or "none"))
    df = df.drop(columns=drop_ids)

    feat = feature_columns(df)
    df[feat] = df[feat].apply(pd.to_numeric, errors="coerce")
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    bad = df[feat].isna().any(axis=1)
    log.append(("rows with NaN or inf removed", int(bad.sum())))
    df = df[~bad]
    # Cast BEFORE deduplicating: rows that differ only beyond float32 precision
    # become identical after the cast and would otherwise leak across the split.
    df[feat] = df[feat].astype(np.float32)

    dup = df.duplicated(subset=feat + ["family"])
    log.append(("exact duplicate rows removed", int(dup.sum())))
    df = df[~dup]

    # Exact duplicates are gone, so any row still sharing its features carries another label.
    conflict = df.duplicated(subset=feat, keep=False)
    log.append(("rows with identical features but different labels removed", int(conflict.sum())))
    df = df[~conflict]

    df["y"] = (df["family"] != C.BENIGN).astype(np.int8)
    log.append(("rows after cleaning", len(df)))
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list:
    skip = {C.LABEL_COL, "family", "y", "source_file"}
    return [c for c in df.columns if c not in skip]


def stratified_sample(df: pd.DataFrame, max_rows=C.MAX_ROWS, keep_all_below=C.KEEP_ALL_BELOW,
                      seed=C.SEED) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    counts = df["family"].value_counts()
    small = counts[counts < keep_all_below].index
    keep_small = df[df["family"].isin(small)]
    rest = df[~df["family"].isin(small)]
    frac = (max_rows - len(keep_small)) / len(rest)
    rest = rest.groupby("family", group_keys=False).sample(frac=frac, random_state=seed)
    return pd.concat([keep_small, rest]).sample(frac=1, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, seed=C.SEED):
    """60/20/20, stratified by family. Families with < 5 rows are stratified by y instead."""
    strat = df["family"].where(df["family"].map(df["family"].value_counts()) >= 5, "rare_" + df["y"].astype(str))
    train, tmp = train_test_split(df, test_size=C.VAL_FRAC + C.TEST_FRAC, stratify=strat, random_state=seed)
    strat_tmp = strat.loc[tmp.index]
    strat_tmp = strat_tmp.where(strat_tmp.map(strat_tmp.value_counts()) >= 2, "rare_" + tmp["y"].astype(str))
    val, test = train_test_split(tmp, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
                                 stratify=strat_tmp, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def select_features(train: pd.DataFrame, log: list) -> list:
    """Learned on TRAIN only: drop constant columns and exact duplicate columns."""
    feat = feature_columns(train)
    const = [c for c in feat if train[c].nunique() <= 1]
    feat = [c for c in feat if c not in const]
    dup_cols = train[feat].T.duplicated()
    dups = list(dup_cols[dup_cols].index)
    feat = [c for c in feat if c not in dups]
    log.append(("constant columns dropped (train)", ", ".join(const) or "none"))
    log.append(("duplicate columns dropped (train)", ", ".join(dups) or "none"))
    log.append(("features kept", len(feat)))
    return feat


def leakage_check(train, val, test, feat) -> pd.DataFrame:
    """Rows of val/test whose feature vector also appears in train (should be 0 after dedup)."""
    key = lambda d: pd.util.hash_pandas_object(d[feat], index=False)
    tr = set(key(train))
    return pd.DataFrame({
        "set": ["val", "test"],
        "rows": [len(val), len(test)],
        "also_in_train": [int(key(val).isin(tr).sum()), int(key(test).isin(tr).sum())],
    })


def resolve_suspects(feat: list, patterns=C.SUSPECT_PATTERNS) -> list:
    return [c for c in feat if any(p in c for p in patterns)]


def drop_train_overlap(train, other, feat):
    """Remove rows of val/test whose (kept) feature vector also appears in train.
    Needed because columns dropped as constant on train can hide duplicates."""
    key = lambda d: pd.util.hash_pandas_object(d[feat], index=False)
    mask = key(other).isin(set(key(train)))
    return other[~mask.values].reset_index(drop=True), int(mask.sum())
