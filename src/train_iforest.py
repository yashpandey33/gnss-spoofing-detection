"""Isolation Forest: learns NORMAL behaviour only, flags anything unusual (no attack labels needed for training).
Run: python -m src.train_iforest [kitti_oxts_folder]"""
import sys
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from src.data import load
from src.attacks import inject
from src.features import make_features

df = inject(load(sys.argv[1] if len(sys.argv) > 1 else None))
X, y = make_features(df), df["label"]
c = int(len(df) * 0.8)                                       # time-based 80/20
Xtr, Xte, ytr, yte = X[:c], X[c:], y[:c], y[c:]

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(Xtr[ytr == 0])  # normal rows only
pred = (iso.predict(Xte) == -1).astype(int)                  # -1 = anomaly -> 1 = attack
print("=== Isolation Forest ===")
print(classification_report(yte, pred, zero_division=0, digits=3))
