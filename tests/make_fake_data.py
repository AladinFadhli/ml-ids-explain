"""Synthetic CSVs shaped like CIC-IDS2017 MachineLearningCVE, to smoke-test the pipeline.
NOT real traffic. Usage: python tests/make_fake_data.py <out_dir> [rows]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

families = {  # name: (share, port, init_win)
    "BENIGN": (0.80, None, None),
    "DoS Hulk": (0.08, 80, 29200),
    "PortScan": (0.06, None, 1024),
    "SSH-Patator": (0.03, 22, 29200),
    "Web Attack \x96 Brute Force": (0.02, 80, 8192),
    "Heartbleed": (0.0004, 444, 29200),
}
rows = []
for fam, (share, port, win) in families.items():
    k = max(int(n * share), 11)
    benign = fam == "BENIGN"
    d = {
        " Destination Port": (rng.choice([53, 80, 443, 8080, 22], k) if benign
                              else rng.integers(1, 65535, k) if fam == "PortScan" else np.full(k, port)),
        " Flow Duration": rng.exponential(1e5 if benign else 3e4, k).astype(int),
        " Total Fwd Packets": rng.poisson(8 if benign else 3, k) + 1,
        " Total Backward Packets": rng.poisson(7 if benign else 2, k),
        "Total Length of Fwd Packets": rng.exponential(900 if benign else 300, k),
        " Fwd Packet Length Mean": rng.normal(120 if benign else 60, 30, k).clip(0),
        " Bwd Packet Length Mean": rng.normal(300 if benign else 90, 80, k).clip(0),
        "Flow Bytes/s": rng.exponential(5e4, k),
        " Flow IAT Mean": rng.exponential(2e4 if benign else 5e3, k),
        " SYN Flag Count": rng.binomial(1, 0.1 if benign else 0.6, k),
        " Bwd PSH Flags": np.zeros(k, int),
        " Fwd Header Length": rng.integers(20, 400, k),
        "Init_Win_bytes_forward": rng.choice([8192, 29200, 65535, -1], k) if benign else np.full(k, win),
        " Init_Win_bytes_backward": rng.choice([-1, 235, 28960], k),
        " Label": fam,
    }
    df = pd.DataFrame(d)
    df[" Fwd Header Length.1"] = df[" Fwd Header Length"]
    rows.append(df)

df = pd.concat(rows).sample(frac=1, random_state=0).reset_index(drop=True)
df.loc[df.sample(40, random_state=1).index, "Flow Bytes/s"] = np.inf
df.loc[df.sample(20, random_state=2).index, "Flow Bytes/s"] = np.nan
df = pd.concat([df, df.sample(500, random_state=3)])          # duplicates
for i, idx in enumerate(np.array_split(np.arange(len(df)), 3)):
    part = df.iloc[idx]
    part.to_csv(out / f"Day{i + 1}-WorkingHours.pcap_ISCX.csv", index=False, encoding="latin-1")
print("wrote", len(df), "rows to", out)
