"""Signal module (Jammertest 2024): Random Forest + Isolation Forest + trust score, same design as src.evaluate.
Run: python -m src.evaluate_signal data/jammertest.csv"""
import sys
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLS = ["cn0", "cn0_spread", "nsat", "cn0_mean5", "cn0_std5"]
d = pd.read_csv(sys.argv[1])
tr, va, te = [], [], []
for _, g in d.groupby("file"):                                       # time-ordered 60/20/20 inside each file
    a, b = int(len(g) * 0.6), int(len(g) * 0.8)
    tr += list(g.index[:a]); va += list(g.index[a:b]); te += list(g.index[b:])
X, y3 = d[COLS], d["label"]                                          # y3: 0 normal, 1 spoofing, 2 jamming
y = (y3 > 0).astype(int)
Xtr, Xva, Xte, ytr, yva, yte = X.loc[tr], X.loc[va], X.loc[te], y.loc[tr], y.loc[va], y.loc[te]
Xn = Xtr[ytr == 0]
if len(Xn) == 0 or ytr.sum() == 0:
    sys.exit("Need both normal and attack rows. Check the class counts printed by signal_prep.")

rf = RandomForestClassifier(200, class_weight="balanced", random_state=0).fit(Xtr, y3.loc[tr])   # 3 classes
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(Xn)
raw = -iso.score_samples(Xn)
m = {"rf": rf, "iso": iso, "lo": np.percentile(raw, 5), "hi": np.percentile(raw, 99.5), "w": (0.5, 0.5), "thr": 50}

def parts(m, X):                                                     # Isolation Forest score S, Random Forest score C
    S = np.clip((-m["iso"].score_samples(X[COLS]) - m["lo"]) / (m["hi"] - m["lo"] + 1e-9), 0, 1)
    C = 1 - m["rf"].predict_proba(X[COLS])[:, list(m["rf"].classes_).index(0)]    # P(attack) = 1 - P(normal)
    return S, C

S, C = parts(m, Xva)
best = (-1, None, None)
for wS in np.arange(0.1, 0.91, 0.1):                                 # weights sum to 1, each >= 0.1
    for thr in (30, 40, 50, 60, 70):
        f1 = f1_score(yva, (100 * (1 - (wS * S + (1 - wS) * C)) < thr).astype(int), zero_division=0)
        if f1 > best[0]: best = (f1, (wS, 1 - wS), thr)
m["w"], m["thr"] = best[1], best[2]
print(f"Chosen on validation: weights (IsolationForest, RandomForest) = {tuple(round(float(v), 1) for v in m['w'])}, alert if trust < {m['thr']}\n")

def row(name, flag, score):
    flag = np.asarray(flag).astype(int)
    try: auc = roc_auc_score(yte, score)
    except ValueError: auc = float("nan")
    return {"model": name, "accuracy": accuracy_score(yte, flag), "precision": precision_score(yte, flag, zero_division=0),
            "recall": recall_score(yte, flag), "f1": f1_score(yte, flag, zero_division=0), "roc_auc": auc,
            "false_alarm_rate": ((flag == 1) & (yte.to_numpy() == 0)).sum() / max(1, (yte == 0).sum())}

mu, sd = Xn[["cn0", "nsat"]].mean(), Xn[["cn0", "nsat"]].std() + 1e-9
s_rule = ((Xte[["cn0", "nsat"]] - mu).abs() / sd).max(axis=1).to_numpy() / 3        # 3-sigma rule on C/N0 and satellites
S, C = parts(m, Xte)
trust = 100 * (1 - (m["w"][0] * S + m["w"][1] * C))
rows = [row("Signal-strength rule (3-sigma)", s_rule > 1, s_rule),
        row("Isolation Forest only", iso.predict(Xte[COLS]) == -1, -iso.score_samples(Xte[COLS])),
        row("Random Forest only", C > 0.5, C),
        row("Isolation + Random Forest (trust score)", trust < m["thr"], (100 - trust) / 100)]
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nRandom Forest, 3 classes (test part):")
print(classification_report(y3.loc[te], rf.predict(Xte), labels=[0, 1, 2], target_names=["normal", "spoofing", "jamming"],
                            zero_division=0, digits=3))
joblib.dump(m, PROJECT_ROOT / "models" / "trust_signal.pkl")
