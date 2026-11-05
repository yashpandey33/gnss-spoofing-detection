# Simulated KITTI/OXTS evaluation notes

## Dataset and method
- Three local OXTS datasets: 1,215 rows total (447 + 660 + 108).
- Each drive was split chronologically into 60% training, 20% validation, and 20% test segments.
- Training spoofing examples used strength multipliers 0.1, 0.25, 0.5, and 1.0 (15, 40, 15, and 30 labeled rows respectively). Validation included 0.1, 0.25, and 0.5 examples (13, 12, and 21 labeled rows).
- Jamming and spoofing events are generated separately inside each split. The held-out test and strength-stress attacks use separate seeds from training; test-seed repetitions use the same source drives.
- Strength values scale the simulated position-offset magnitude; they are not calibrated physical attack levels.

## Primary held-out test
- Rows: 244 total; normal 162, spoofing 46, jamming 36.
- Validation-selected trust weights (Isolation Forest, Random Forest, residual): 0.1, 0.8, 0.1; alert threshold: trust < 40.
- Trust-score detector: precision 0.788, recall 0.634, F1 0.703, ROC AUC 0.925, false-alarm rate 0.086.
- Random Forest spoofing class: precision 0.744, recall 0.630, F1 0.682 (46 spoofing rows).
- Random Forest jamming class: precision 1.000, recall 0.944, F1 0.971 (36 jamming rows).
- Spoofing events correctly classified: 4/4; jamming events: 4/4.
- Trust alerts detected 7/8 attack events; mean alert delay was 0.075 seconds.

## Five-seed test variability
- Trust-score binary F1: 0.722 +/- 0.065; false-alarm rate: 0.066 +/- 0.015.
- Spoofing classification precision: 0.782 +/- 0.046; recall: 0.679 +/- 0.072; F1: 0.726 +/- 0.059.
- Jamming classification precision: 1.000 +/- 0.000; recall: 0.886 +/- 0.055; F1: 0.939 +/- 0.031.
- Spoofing event classification recall: 0.900 +/- 0.137.
- Trust-alert spoofing row recall by profile: step 0.371 +/- 0.131; linear drift 0.697 +/- 0.369; smooth drift 0.803 +/- 0.155; oscillating drift 0.661 +/- 0.188.

## Spoofing-strength stress test and comparison

| Strength | Row recall before | Row recall after | Events before | Events after | False-alarm rate after |
|---|---:|---:|---:|---:|---:|
| 0.1 | 0.004 | 0.049 | 2/13 | 4/13 | 0.043 |
| 0.25 | 0.041 | 0.275 | 5/13 | 9/13 | 0.074 |
| 0.5 | 0.384 | 0.727 | 11/13 | 13/13 | 0.074 |
| 1.0 | 0.799 | 0.818 | 13/13 | 13/13 | 0.074 |

Weak-attack recall improved in this synthetic stress test, especially at strength 0.25. However, the false-alarm rate increased, and primary held-out trust-score F1 decreased from 0.768 to 0.703. Across five test seeds, mean spoofing recall rose from 0.642 to 0.679 while the mean false-alarm rate rose from 0.011 to 0.066. This is a sensitivity/false-alarm tradeoff, not an overall improvement or evidence of real-world detection.

## Limitations
- These are simulated attacks, not real spoofing or jamming recordings. Results describe the chosen synthetic patterns only.
- Training on weak examples improved the weak-strength stress-test recall, but it increased false alarms; subtle spoofing remains difficult to identify reliably.
- The time split is within the same source drives, not a held-out-drive evaluation. Test-seed repetitions do not change that.
- KITTI/OXTS does not provide raw C/N0 or GNSS observables; satellite count and position accuracy are only proxies.
- Alerts are indicators, not proof. Independent routes and real labeled interference data are needed before making operational claims.
