"""Convert Jammertest RINEX observation files -> one CSV for training.
Run: python -m src.signal_prep <extracted_folder> [data/jammertest.csv]
Labels come from words in the file path (see guess). Optional data/windows.csv
(file_keyword,start,end,label) marks exact attack time windows."""
import os, sys, warnings
from pathlib import Path

import numpy as np, pandas as pd, georinex as gr
from src.signal_features import add_rolling

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def is_obs(name):
    n = name.lower()
    return n.endswith((".obs", ".rnx", ".crx")) or (len(n) > 4 and n[-4] == "." and n[-3:-1].isdigit() and n[-1] == "o")

def guess(path):                                   # 0 normal, 1 spoofing(+meaconing), 2 jamming, -1 unknown
    p = path.lower()
    if any(k in p for k in ("clean", "nominal", "baseline", "normal")): return 0
    if any(k in p for k in ("spoof", "meacon")): return 1
    if "jam" in p: return 2
    return -1

def rinex_to_df(path):
    ds = gr.load(path)
    s_vars = [v for v in ds.data_vars if v.startswith("S")]      # S* = signal strength (C/N0)
    if not s_vars: raise ValueError("no S* (signal strength) columns")
    a = ds[s_vars].to_array().values                             # (obs type, time, satellite)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sat_cn0 = np.nanmean(a, axis=0)                          # (time, satellite)
        df = pd.DataFrame({"time": pd.to_datetime(ds.time.values),
                           "cn0": np.nanmean(sat_cn0, axis=1),
                           "cn0_spread": np.nanstd(sat_cn0, axis=1),
                           "nsat": np.sum(~np.isnan(sat_cn0), axis=1)})
    df["cn0"] = df["cn0"].fillna(0)                              # epochs with no satellites = signal lost
    df["cn0_spread"] = df["cn0_spread"].fillna(0)
    return add_rolling(df)

def label_rows(df, path, windows):
    df["label"] = guess(path)
    if windows is not None:
        w = windows[windows["file_keyword"].apply(lambda k: k.lower() in path.lower())]
        if len(w):
            df["label"] = 0
            for _, r in w.iterrows():
                m = (df["time"] >= pd.Timestamp(r["start"])) & (df["time"] <= pd.Timestamp(r["end"]))
                df.loc[m, "label"] = int(r["label"])
    return df

if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve() if Path(sys.argv[1]).is_absolute() else (PROJECT_ROOT / sys.argv[1])
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 and Path(sys.argv[2]).is_absolute() else (PROJECT_ROOT / (sys.argv[2] if len(sys.argv) > 2 else "data/jammertest.csv"))
    windows_path = PROJECT_ROOT / "data" / "windows.csv"
    windows = pd.read_csv(windows_path) if windows_path.exists() else None
    parts = []
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if not is_obs(f): continue
            path = os.path.join(d, f)
            try:
                df = label_rows(rinex_to_df(path), path, windows)
            except Exception as e:
                print(f"SKIP {path}: {e}"); continue
            if (df["label"] < 0).all():
                print(f"SKIP {path}: cannot guess label from the path (add it to data/windows.csv)"); continue
            df = df[df["label"] >= 0]
            df["file"] = path
            parts.append(df); print(f"OK   {path}: {len(df)} rows, label counts {df['label'].value_counts().to_dict()}")
    if not parts: sys.exit("No usable RINEX files found. Run src.explore first.")
    all_ = pd.concat(parts, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_.to_csv(out, index=False)
    print(f"\nSaved {out}: {len(all_)} rows\n", all_["label"].value_counts().rename({0: "normal", 1: "spoofing", 2: "jamming"}))
