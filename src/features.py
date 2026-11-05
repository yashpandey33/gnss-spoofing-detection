import numpy as np, pandas as pd

FEATURES = ["gps_speed", "imu_speed", "diff", "jump", "diff_mean", "diff_std",
            "residual", "numsats", "navstat", "pos_accuracy"]

def make_features(df, dt=0.1, win=5, rwin=10):
    f = pd.DataFrame(index=df.index)
    cos = np.cos(np.radians(df["lat"]))
    dy = df["lat"].diff() * 111320
    dx = df["lon"].diff() * 111320 * cos
    f["jump"] = np.hypot(dx, dy)                         # metres moved per step
    f["gps_speed"] = f["jump"] / dt
    f["imu_speed"] = df["imu_speed"]
    f["diff"] = f["gps_speed"] - f["imu_speed"]          # GPS vs IMU speed mismatch
    f["diff_mean"] = f["diff"].rolling(win).mean()
    f["diff_std"] = f["diff"].rolling(win).std()
    gn = df["lat"].diff(rwin) * 111320                   # GPS displacement over last 10 steps
    ge = df["lon"].diff(rwin) * 111320 * cos
    inn = (df["vn"] * dt).rolling(rwin).sum()            # IMU dead-reckoned displacement
    ine = (df["ve"] * dt).rolling(rwin).sum()
    f["residual"] = np.hypot(gn - inn, ge - ine)         # GPS-IMU residual (metres)
    f["numsats"], f["navstat"], f["pos_accuracy"] = df["numsats"], df["navstat"], df["pos_accuracy"]
    return f.fillna(0)[FEATURES]
