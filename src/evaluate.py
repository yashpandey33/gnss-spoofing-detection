"""Full evaluation from the synopsis: Random Forest + Isolation Forest + trust score, compared with 3 simpler detectors.
Run: python -m src.evaluate "C:\\path\\drive1\\oxts\\data" ["C:\\path\\drive2\\oxts\\data" ...]   (no path = built-in sample data)
Several drives are processed one by one (no jumps between drives) and their rows are pooled."""
import sys, itertools
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from src.data import discover_kitti_dirs, load
from src.attacks import inject, inject_balanced
from src.features import make_features
from src.metrics import detection_delay, event_recall
from src.trust import components, trust_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = sys.argv[1:] or discover_kitti_dirs(PROJECT_ROOT) or [None]
GPS_ONLY = ["gps_speed", "jump", "numsats", "navstat", "pos_accuracy"]     # what the GPS receiver alone gives

drives = [(p, load(p)) for p in PATHS]
for p, d in drives:
    if len(d) < 60: sys.exit(f"Drive {p} has only {len(d)} rows. Remove it from the command.")
total = sum(len(d) for _, d in drives)
print(f"{len(drives)} drive(s), {total} rows in total")
if total < 500:
    print("WARNING: fewer than 500 rows in total. Add more drives for meaningful results.\n")

parts = {"tr": [], "va": []}                                        # time-ordered 60 / 20 / 20 inside EACH drive
spoof_strengths = {"tr": [], "va": []}
for i, (_, base) in enumerate(drives):
    a, b = int(len(base) * 0.6), int(len(base) * 0.8)
    for split, start, end in (("tr", 0, a), ("va", a, b)):
        segment = inject_balanced(
            base.iloc[start:end],
            seed=i * 10 + (1 if split == "tr" else 2),
            spoof_strengths=(0.1, 0.25, 0.5, 1.0),
        )
        parts[split].append((make_features(segment), segment["label"], segment["attack"]))
        spoof_strengths[split].append(segment.loc[segment["attack"] == 1, "spoof_strength"])
cat = lambda k, j: pd.concat([p[j] for p in parts[k]], ignore_index=True)
Xtr, ytr, atr = cat("tr", 0), cat("tr", 1), cat("tr", 2)
Xva, yva, ava = cat("va", 0), cat("va", 1), cat("va", 2)
strength_counts = {
    split: pd.concat(values).value_counts().sort_index()
    for split, values in spoof_strengths.items()
}
print("Spoofing strength multipliers (training):", strength_counts["tr"].to_dict())
print("Spoofing strength multipliers (validation):", strength_counts["va"].to_dict())
for split_name, labels in (("training", atr), ("validation", ava)):
    missing = sorted(set((0, 1, 2)) - set(labels.unique()))
    if missing:
        sys.exit(f"The {split_name} split is missing attack classes {missing}. Use longer drives.")
Xn = Xtr[ytr == 0]                                                  # normal training rows

rf = RandomForestClassifier(200, class_weight="balanced", random_state=0).fit(Xtr, atr)       # 3 classes
rf_gps = RandomForestClassifier(200, class_weight="balanced", random_state=0).fit(Xtr[GPS_ONLY], ytr)
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(Xn)

raw = -iso.score_samples(Xn)
demo = max(drives, key=lambda t: len(t[1]))[0]                      # longest drive is shown on the dashboard
m = {"rf": rf, "iso": iso, "demo": demo, "datasets": [str(p) for p, _ in drives],
     "training_rows": total, "w": (1 / 3, 1 / 3, 1 / 3), "thr": 50,
     "lo": np.percentile(raw, 5), "hi": np.percentile(raw, 99.5),
     "r50": np.percentile(Xn["residual"], 50), "r99": np.percentile(Xn["residual"], 99)}

# choose weights (sum = 1) and alert threshold on the VALIDATION part only
comp_va, best = components(m, Xva), (-1, None, None)
for wS, wC in itertools.product(np.arange(0.1, 0.81, 0.1), repeat=2):   # every part keeps weight >= 0.1
    if wS + wC > 0.9001: continue
    w = (wS, wC, 1 - wS - wC)
    t = 100 * (1 - (w[0] * comp_va[0] + w[1] * comp_va[1] + w[2] * comp_va[2]))
    for thr in (30, 40, 50, 60, 70):
        f1 = f1_score(yva, (t < thr).astype(int), zero_division=0)
        if f1 > best[0]: best = (f1, w, thr)
m["w"], m["thr"] = best[1], best[2]
print(f"Chosen on validation: weights (IsolationForest, RandomForest, residual) = {tuple(round(float(v), 1) for v in m['w'])}, alert if trust < {m['thr']}\n")

def row(name, truth, flag, score):
    flag = np.asarray(flag).astype(int)
    truth = np.asarray(truth).astype(int)
    try:
        auc = roc_auc_score(truth, score)
    except ValueError:
        auc = float("nan")
    return {"model": name, "accuracy": accuracy_score(truth, flag), "precision": precision_score(truth, flag, zero_division=0),
            "recall": recall_score(truth, flag, zero_division=0), "f1": f1_score(truth, flag, zero_division=0), "roc_auc": auc,
            "false_alarm_rate": ((flag == 1) & (truth == 0)).sum() / max(1, (truth == 0).sum())}

def make_test_set(seed):
    chunks = []
    for i, (_, base) in enumerate(drives):
        start, end = int(len(base) * 0.8), len(base)
        segment = inject_balanced(base.iloc[start:end], seed=seed + i * 37)
        chunks.append((make_features(segment), segment["label"], segment["attack"], segment["scenario"]))
    return chunks


def event_class_recall(chunks, predictions, class_id):
    detected = total_events = offset = 0
    for _, _, attacks, _ in chunks:
        length = len(attacks)
        hits, count = event_recall(attacks.to_numpy() == class_id,
                                   predictions[offset:offset + length] == class_id)
        detected += hits
        total_events += count
        offset += length
    return detected, total_events


def evaluate_test(chunks):
    Xte = pd.concat([chunk[0] for chunk in chunks], ignore_index=True)
    yte = pd.concat([chunk[1] for chunk in chunks], ignore_index=True)
    ate = pd.concat([chunk[2] for chunk in chunks], ignore_index=True)
    r99, d99 = np.percentile(Xn["residual"], 99), np.percentile(Xn["diff"].abs(), 99)
    s_imu = Xte["residual"].to_numpy() / r99
    s_fus = np.maximum(s_imu, Xte["diff"].abs().to_numpy() / d99)
    p_gps = rf_gps.predict_proba(Xte[GPS_ONLY])[:, 1]
    t_fin = trust_score(m, Xte)
    flag_fin = t_fin < m["thr"]
    y_binary = (yte > 0).to_numpy()
    rows = [row("GPS-only ML (Random Forest)", y_binary, p_gps > 0.5, p_gps),
            row("IMU dead-reckoning check", y_binary, s_imu > 1, s_imu),
            row("GPS + IMU rule", y_binary, s_fus > 1, s_fus),
            row("GPS + IMU + ML (final, trust score)", y_binary, flag_fin, (100 - t_fin) / 100)]
    predicted_classes = rf.predict(Xte).astype(int)
    report = classification_report(ate, predicted_classes, labels=[0, 1, 2],
                                   target_names=["normal", "spoofing", "jamming"],
                                   zero_division=0, digits=3, output_dict=True)
    cm = confusion_matrix(ate, predicted_classes, labels=[0, 1, 2])
    binary_events = []
    offset = 0
    for _, labels, _, _ in chunks:
        length = len(labels)
        binary_events.append(event_recall((labels > 0).to_numpy(), flag_fin[offset:offset + length]))
        offset += length
    class_events = {class_id: event_class_recall(chunks, predicted_classes, class_id) for class_id in (1, 2)}
    normal_rows = yte.to_numpy() == 0
    summary = {
        "f1": float(f1_score(y_binary, flag_fin, zero_division=0)),
        "false_alarm_rate": float((flag_fin & normal_rows).sum() / max(1, normal_rows.sum())),
        "spoofing_precision": float(report["spoofing"]["precision"]),
        "spoofing_recall": float(report["spoofing"]["recall"]),
        "spoofing_f1": float(report["spoofing"]["f1-score"]),
        "jamming_precision": float(report["jamming"]["precision"]),
        "jamming_recall": float(report["jamming"]["recall"]),
        "jamming_f1": float(report["jamming"]["f1-score"]),
        "spoofing_event_recall": class_events[1][0] / max(1, class_events[1][1]),
    }
    return {
        "X": Xte, "labels": yte, "attacks": ate, "trust": t_fin, "alert": flag_fin,
        "predicted_classes": predicted_classes, "rows": rows, "report": report, "matrix": cm,
        "binary_events": binary_events, "class_events": class_events, "summary": summary,
        "chunks": chunks,
    }


test_seeds = (1000, 2001, 3002, 4003, 5000)
test_runs = [evaluate_test(make_test_set(seed)) for seed in test_seeds]
primary = test_runs[0]
Xte, yte, ate = primary["X"], primary["labels"], primary["attacks"]
rows = primary["rows"]
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nRandom Forest, 3 classes (test part):")
class_report = primary["report"]
print(classification_report(ate, primary["predicted_classes"], labels=[0, 1, 2],
                            target_names=["normal", "spoofing", "jamming"],
                            zero_division=0, digits=3))
print("Confusion matrix (rows=true, columns=predicted; normal, spoofing, jamming):")
print(primary["matrix"])

cuts = np.cumsum([len(chunk[1]) for chunk in primary["chunks"]])[:-1]
delays = [detection_delay(y, f, dt=0.1)
          for y, f in zip(np.split(yte.to_numpy(), cuts), np.split(primary["alert"], cuts))]
delays = [x for x in delays if x is not None]
print("Average detection delay of final system (seconds):", float(np.mean(delays)) if delays else None)
ev = primary["binary_events"]
print(f"Attacks detected by trust alert (at least one alert during event): {sum(e[0] for e in ev)} of {sum(e[1] for e in ev)}")
for class_id, class_name in ((1, "Spoofing"), (2, "Jamming")):
    hit, total_events = primary["class_events"][class_id]
    print(f"{class_name} events correctly classified by Random Forest: {hit} of {total_events}")
for class_id, class_name in ((1, "spoofing"), (2, "jamming")):
    print(f"{class_name.title()} test rows: {int((ate == class_id).sum())}")

print(f"\nRepeated held-out synthetic attack tests ({len(test_runs)} independent seeds; same source drives):")
print("metric                   mean     std")
for metric in ("f1", "false_alarm_rate", "spoofing_precision", "spoofing_recall", "spoofing_f1",
               "jamming_precision", "jamming_recall", "jamming_f1", "spoofing_event_recall"):
    values = np.array([run["summary"][metric] for run in test_runs])
    print(f"{metric:<24} {values.mean():.3f}    {values.std(ddof=1):.3f}")

profile_metrics = {}
for profile in ("step", "linear_drift", "smooth_drift", "oscillating_drift"):
    scored = []
    for run in test_runs:
        mask = run["chunks"][0][3].eq(profile).to_numpy()
        for chunk in run["chunks"][1:]:
            mask = np.concatenate((mask, chunk[3].eq(profile).to_numpy()))
        if mask.any():
            truth = run["attacks"].to_numpy()[mask] == 1
            preds = run["alert"][mask]
            scored.append(float(recall_score(truth, preds, zero_division=0)))
    profile_metrics[profile] = {
        "spoofing_row_recall_mean": float(np.mean(scored)) if scored else None,
        "spoofing_row_recall_std": float(np.std(scored, ddof=1)) if len(scored) > 1 else 0.0,
    }
print("\nTrust-alert recall by simulated spoofing profile (mean +/- std across seeds):")
for profile, values in profile_metrics.items():
    mean_recall = values["spoofing_row_recall_mean"]
    result = f"{mean_recall:.3f} +/- {values['spoofing_row_recall_std']:.3f}" if mean_recall is not None else "no test samples"
    print(f"{profile:<20} {result}")

m["evaluation"] = {
    "training_rows": int(len(Xtr)),
    "validation_rows": int(len(Xva)),
    "test_rows": int(len(Xte)),
    "test_seed_count": len(test_runs),
    "test_seed_values": list(test_seeds),
    "weights": [float(v) for v in m["w"]],
    "threshold": float(m["thr"]),
    "detectors": [{key: (None if pd.isna(value) else float(value)) if key != "model" else value
                   for key, value in item.items()} for item in rows],
    "classes": {
        name: {
            "precision": float(class_report[name]["precision"]),
            "recall": float(class_report[name]["recall"]),
            "f1": float(class_report[name]["f1-score"]),
            "support": int(class_report[name]["support"]),
            "predicted_support": int(primary["matrix"][:, class_id].sum()),
        }
        for class_id, name in enumerate(("normal", "spoofing", "jamming"))
    },
    "class_confusion_matrix": primary["matrix"].tolist(),
    "attack_events_detected": int(sum(e[0] for e in ev)),
    "attack_events_total": int(sum(e[1] for e in ev)),
    "class_events": {
        name: {"detected": int(primary["class_events"][class_id][0]),
               "total": int(primary["class_events"][class_id][1])}
        for class_id, name in ((1, "spoofing"), (2, "jamming"))
    },
    "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
    "repeated_test_metrics": {
        metric: {"mean": float(np.mean([run["summary"][metric] for run in test_runs])),
                 "std": float(np.std([run["summary"][metric] for run in test_runs], ddof=1))}
        for metric in test_runs[0]["summary"]
    },
    "spoofing_profile_recall": profile_metrics,
    "training_spoof_strength_counts": {
        str(float(strength)): int(count) for strength, count in strength_counts["tr"].items()
    },
    "synthetic_evaluation_only": True,
}

print("\nSpoofing strength stress test (independent simulated jumps/drifts):")
print("strength  row_recall  events_detected  false_alarm_rate")
for s in (0.1, 0.25, 0.5, 1.0):
    ds = [inject(base, seed=99 + i, strength=s, kinds=("jump", "drift")) for i, (_, base) in enumerate(drives)]
    X2 = pd.concat([make_features(d) for d in ds], ignore_index=True)
    y2 = np.concatenate([d["label"].to_numpy() for d in ds])
    fl = trust_score(m, X2) < m["thr"]
    cut2 = np.cumsum([len(d) for d in ds])[:-1]
    e2 = [event_recall(y, f) for y, f in zip(np.split(y2, cut2), np.split(fl, cut2))]
    print(f"{s:<9} {recall_score(y2, fl):.3f}      {sum(e[0] for e in e2)} of {sum(e[1] for e in e2)}          {((fl) & (y2 == 0)).sum() / max(1, (y2 == 0).sum()):.3f}")

joblib.dump(m, PROJECT_ROOT / "models" / "trust.pkl")