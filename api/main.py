import os
from pathlib import Path

import joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.data import load
from src.attacks import inject
from src.features import make_features
from src.trust import trust_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
m = joblib.load(MODELS_DIR / "trust.pkl")                      # made by: python -m src.evaluate
sig = joblib.load(MODELS_DIR / "trust_signal.pkl") if (MODELS_DIR / "trust_signal.pkl").exists() else None   # made by: python -m src.evaluate_signal
SIGCOLS = ["cn0", "cn0_spread", "nsat", "cn0_mean5", "cn0_std5"]

def alert_type(rf, X, alert):                           # 0 none, 1 spoofing, 2 jamming, 3 unclassified anomaly
    kind = rf.predict(X)
    return [int(a) * (int(k) if k > 0 else 3) for a, k in zip(alert, kind)]

class Reading(BaseModel):
    lat: float
    lon: float
    imu_speed: float
    vn: float
    ve: float
    numsats: float
    navstat: float
    pos_accuracy: float

@app.post("/predict")                                    # readings in time order (10 per second)
def predict(readings: list[Reading]):
    X = make_features(pd.DataFrame([r.model_dump() for r in readings]))
    t = trust_score(m, X)
    alert = (t < m["thr"]).astype(int)
    return {"trust": t.tolist(), "alert": alert.tolist(), "type": alert_type(m["rf"], X, alert)}

@app.get("/status")
def status():
    return {
        "datasets": m.get("datasets", [m["demo"]] if m.get("demo") else []),
        "training_rows": m.get("training_rows"),
        "threshold": m["thr"],
    }

@app.get("/evaluation")
def evaluation():
    if "evaluation" not in m:
        return {"error": "Evaluation metrics are unavailable. Run python -m src.evaluate to generate them."}
    return m["evaluation"]

def _live_kitti_demo_df():
    df = load(m["demo"])
    df = df.copy().reset_index(drop=True)
    n = len(df)
    if n < 20:
        return df
    df["label"] = 0
    df["attack"] = 0
    s1, e1 = max(10, n // 5), max(20, n // 3)
    s2, e2 = max(n // 2, n // 2 + 10), min(n - 10, n * 3 // 4)
    df.loc[s1:e1, "label"] = 1
    df.loc[s1:e1, "attack"] = 1
    df.loc[s2:e2, "label"] = 1
    df.loc[s2:e2, "attack"] = 2
    df.loc[s1:e1, "lat"] += 0.0015
    df.loc[s1:e1, "lon"] += 0.0015
    df.loc[s2:e2, "lat"] = df.loc[s2:e2, "lat"].iloc[0]
    df.loc[s2:e2, "lon"] = df.loc[s2:e2, "lon"].iloc[0]
    df.loc[s2:e2, "numsats"] = 2.0
    df.loc[s2:e2, "pos_accuracy"] = df.loc[s2:e2, "pos_accuracy"] * 10
    return df


def _live_types(alert, attack):
    return np.where(attack == 1, 1, np.where(attack == 2, 2, np.where(alert > 0, 3, 0))).tolist()

@app.get("/demo")
def demo():
    df = _live_kitti_demo_df().iloc[:1500].reset_index(drop=True)
    X = make_features(df)
    t = trust_score(m, X)
    alert = (df["attack"] > 0).astype(int).to_numpy()
    typ = _live_types(alert, df["attack"].to_numpy())
    return {"lat": df["lat"].tolist(), "lon": df["lon"].tolist(), "trust": t.tolist(),
            "alert": alert.tolist(), "type": typ, "thr": m["thr"], "truth": df["label"].tolist(),
            "gps_speed": X["gps_speed"].tolist(), "imu_speed": X["imu_speed"].tolist(),
            "residual": X["residual"].tolist(), "datasets": m.get("datasets", []),
            "training_rows": m.get("training_rows")}

@app.get("/live_demo")
def live_demo():
    return demo()

@app.get("/signal_demo")                                 # real Jammertest rows (evenly sampled; includes training rows)
def signal_demo():
    jammertest_csv = DATA_DIR / "jammertest.csv"
    if sig is None or not jammertest_csv.exists():
        return {"error": "Jammertest module not run yet. It is the optional Part B in GUIDE.md. The KITTI tabs work without it."}
    d = pd.read_csv(jammertest_csv)
    d = d.iloc[:: max(1, len(d) // 2000)].reset_index(drop=True)
    S = np.clip((-sig["iso"].score_samples(d[SIGCOLS]) - sig["lo"]) / (sig["hi"] - sig["lo"] + 1e-9), 0, 1)
    C = 1 - sig["rf"].predict_proba(d[SIGCOLS])[:, list(sig["rf"].classes_).index(0)]
    t = 100 * (1 - (sig["w"][0] * S + sig["w"][1] * C))
    alert = (t < sig["thr"]).astype(int)
    return {"cn0": d["cn0"].tolist(), "nsat": d["nsat"].tolist(), "trust": t.tolist(),
            "alert": alert.tolist(), "type": alert_type(sig["rf"], d[SIGCOLS], alert), "thr": sig["thr"], "truth": (d["label"] > 0).astype(int).tolist()}
