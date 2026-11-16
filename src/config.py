"""Project-wide settings. Change values here, never inside the notebook cells."""
import os
from pathlib import Path

SEED = 42
STABILITY_SEEDS = [42, 7, 2026]          # task 6: re-split + retrain with each seed

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("IDS_DATA_DIR", ROOT / "data" / "raw"))   # put the CIC-IDS2017 CSV files here
FIG_DIR = ROOT / "reports" / "figures"

# --- sampling (CPU budget) -------------------------------------------------
# Proportional stratified sample by attack family, after deduplication.
# Families smaller than KEEP_ALL_BELOW rows are kept whole so rare attacks survive.
MAX_ROWS = 400_000
KEEP_ALL_BELOW = 2_000
TUNE_MAX_ROWS = 60_000       # CV tuning runs on a stratified subset of train
CV_FOLDS = 5

# --- split -----------------------------------------------------------------
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20

# --- explainability --------------------------------------------------------
SHAP_MAX_ROWS = 2_000        # cahier des charges: SHAP on at most 2,000 flows
PERM_REPEATS = 5
PERM_MAX_ROWS = 20_000       # permutation importance on a stratified test subset
TOP_K = 15

# Columns that identify a flow instead of describing its behaviour.
# Normalised names (lower case, non-alphanumerics -> "_").
ID_COLUMNS = [
    "flow_id", "source_ip", "src_ip", "destination_ip", "dst_ip",
    "source_port", "src_port", "timestamp", "unnamed_0", "id",
]

# Task 4 suspects, matched as substrings of normalised column names.
# Edit after the plausibility audit.
SUSPECT_PATTERNS = ["destination_port", "dst_port", "init_win", "init_fwd_win", "init_bwd_win"]

LABEL_COL = "label"
BENIGN = "BENIGN"
