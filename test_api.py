import requests
from src.data import load
from src.attacks import inject

COLS = ["lat", "lon", "imu_speed", "vn", "ve", "numsats", "navstat", "pos_accuracy"]
df = inject(load(), seed=3).iloc[:1500]
r = requests.post("http://127.0.0.1:8000/predict", json=df[COLS].to_dict("records")).json()
hit = sum(a > 0 and t > 0 for a, t in zip(r["alert"], df["label"]))
print(f"attack points: {df['label'].sum()}  alerted: {hit}  false alerts: {sum(a > 0 and t == 0 for a, t in zip(r['alert'], df['label']))}")
