import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.data import load
from src.attacks import inject
from src.features import make_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = inject(load(sys.argv[1] if len(sys.argv) > 1 else None))
X, y = make_features(df), df["label"]
cut = int(len(df) * 0.8)                                  # time-based split (no leakage)
Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

rf = RandomForestClassifier(200, class_weight="balanced", random_state=0).fit(Xtr, ytr)
print("=== Random Forest ===")
print(classification_report(yte, rf.predict(Xte), digits=3))
joblib.dump(rf, PROJECT_ROOT / "models" / "rf.pkl")
