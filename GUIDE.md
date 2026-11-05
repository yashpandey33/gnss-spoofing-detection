# AI-Based GNSS Spoofing & Jamming Detection Using GPS-IMU Fusion — project guide

Run every command from this folder in a Command Prompt with the venv active.

## The plan in one line
Part A (KITTI) is required and gives a complete project. Part B (Jammertest) is planned but optional:
do it only after Part A is finished. If Part B is not working after ONE week, drop it and submit Part A.

TESTED: balanced simulated evaluation and attack-generation tests on the three local OXTS datasets (1,215 rows).
NOT TESTED: real labeled GNSS spoofing/jamming data or the real Jammertest files. Fix small things on first run.

---
# PART A — KITTI (required)
A1. Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
A2. Download a LONG KITTI drive (500+ rows), e.g. 2011_09_26_drive_0036 (synced+rectified). Copy the real path of its
    `oxts\data` folder from the File Explorer address bar.
A3. Run the evaluation (your main result)
```
python -m src.evaluate "PASTE_YOUR_KITTI_PATH_HERE"
```
    Prints weights, 4-detector table, 3-class report and confusion matrix, per-class event detection,
    five independent synthetic test-seed results, profile/strength stress tests, and detection delay.
    Spoofing and jamming events are balanced within each time split. Saves models/trust.pkl. Screenshot.
A4. Dashboard: Terminal 1 `uvicorn api.main:app --reload`; Terminal 2 `python test_api.py`; then double-click `frontend/index.html`.
    Tabs: Map, GPS vs IMU, Trust Score. (Signal tab says "not run yet" until Part B is done. That is normal.)
A5. Report: synopsis sections + screenshots + results table. Part A is a complete submission.

---
# PART B — Jammertest 2024 (optional, only after Part A is finished)
B1. Extra install
```
pip install -r requirements-jammertest.txt
```
B2. Download from https://zenodo.org/records/15911589 the file GNSS_DATASET_JAMMING_SPOOFING.tar.gz (375.3 MB) and README.md. Read the README.
```
mkdir jt
tar -xzf GNSS_DATASET_JAMMING_SPOOFING.tar.gz -C jt
python -m src.explore jt
```
B3. Convert to a table
```
python -m src.signal_prep jt data/jammertest.csv
```
    Success: lines `OK ...` and all THREE classes (normal, spoofing, jamming) with counts.
B4. Signal module
```
python -m src.evaluate_signal data/jammertest.csv
```
    Prints weights, 4-detector table, 3-class report. Saves models/trust_signal.pkl. Screenshot.
B5. Restart the API (Ctrl+C, then `uvicorn api.main:app --reload`) and refresh the dashboard: the Signal tab now shows real data.
B6. Add to the report: a second results table (Jammertest, real attacks), kept separate from KITTI.
    Add ONE sentence to the synopsis: "Real recorded interference from Jammertest 2024 validates a signal-quality module."

DECISION RULE: if B3 does not give all three classes within one week, stop Part B. Submit Part A and list Jammertest as future work.

---
## How alerts are shown
When a trust score falls below its threshold the dashboard shows a red ALERT box listing each event with its type
(spoofing, jamming, or unclassified anomaly) and the point range. Map: red = spoofing, purple = jamming,
orange = unclassified anomaly, yellow = suspicious, green = trusted. The type comes from the Random Forest (3 classes);
"unclassified" means the Isolation Forest found something unusual that the Random Forest calls normal.

## If Part B fails
- `no S* (signal strength) columns` -> that RINEX has no signal strength; use other files.
- `cannot guess label from the path` -> create `data/windows.csv` (columns: file_keyword,start,end,label; label 0 normal, 1 spoofing/meaconing, 2 jamming;
  times from the Jammertest transmission plan; RINEX uses GPS time, GPS = UTC + 18 s in 2024, Norway summer = UTC + 2).
- Only `.ubx` files -> convert to RINEX with RTKLIB `convbin`; `.crx` files -> `pip install hatanaka`.
- `Need both normal and attack rows` -> you need clean + attack files; check the class counts.
- Mixed constellations error -> in `src/signal_prep.py` change `gr.load(path)` to `gr.load(path, use='G')`.
- `netcdf4` will not install -> remove that line from requirements-jammertest.txt and retry.

## Other errors
- `No .txt files in ...` -> use the real `oxts\data` path, not the words PASTE_YOUR_KITTI_PATH_HERE.
- `No module named 'src'` -> run from the project folder with `python -m ...`.
- Dashboard "API error" -> run A3 first, then start the API.

## Map to the synopsis
| Synopsis | Code |
|---|---|
| Data + simulated attacks (training includes 0.1x, 0.25x, 0.5x, and 1.0x spoof strengths) | `src/data.py`, `src/attacks.py` |
| Features (GPS jump, velocity, satellites, fix, accuracy, GPS-IMU residual) | `src/features.py` |
| Isolation Forest + Random Forest | `src/evaluate.py` (Part B: `src/evaluate_signal.py`) |
| Trust score 0-100 + alert | `src/trust.py` |
| Dashboard | `api/main.py`, `frontend/index.html` |
| Evaluation + detection delay + attack strength | `src/evaluate.py` |

## Honest limits (write in the report)
- KITTI attacks are simulated by our code; Jammertest attacks (Part B) are real. Never merge the two result tables.
- KITTI split is time-ordered (60/20/20) inside each drive, not trajectory-level. Say "time-ordered split".
- The five test-seed repetitions vary generated attacks on the same source drives; they do not represent five independent real-world datasets.
- Training includes balanced simulated jump/drift profiles at 0.1x, 0.25x, 0.5x, and 1.0x strengths; validation includes weak-strength examples, while held-out tests use separate attack seeds. This improves simulated weak-attack coverage, not realism. Report per-class support and metrics; do not infer real spoofing performance from simulated results.
- KITTI has no C/N0; satellite count and position accuracy are proxies there.
- In the simulation, jamming also lowers satellite count and accuracy (assumption). Spoofing leaves them normal.
- Weak spoofing (small offsets) is hard to catch: see the attack-strength table.
- The two trust scores are separate; the lower-of-both rule (`fuse` in `src/trust.py`) needs a live receiver and is not shown.
- 100% on Jammertest = suspicious, not a success.
- Alerts are indicators, not proof. Detection happens during an attack, not before.
