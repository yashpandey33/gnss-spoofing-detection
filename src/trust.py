import numpy as np
from src.features import FEATURES

def components(m, X):
    """Three scores, each 0..1 (1 = very suspicious): Isolation Forest, Random Forest, GPS-IMU residual."""
    X = X.reindex(columns=FEATURES)
    S = np.clip((-m["iso"].score_samples(X) - m["lo"]) / (m["hi"] - m["lo"] + 1e-9), 0, 1)
    C = 1 - m["rf"].predict_proba(X)[:, list(m["rf"].classes_).index(0)]
    R = np.clip((X["residual"].to_numpy() - m["r50"]) / (m["r99"] - m["r50"] + 1e-9), 0, 1)
    return S, C, R

def trust_score(m, X, w=None):
    """Trust = 100 * (1 - (wS*S + wC*C + wR*R)), weights sum to 1. 100 = fully trusted, 0 = surely attacked."""
    w = w or m["w"]
    S, C, R = components(m, X)
    return 100 * (1 - (w[0] * S + w[1] * C + w[2] * R))

def fuse(trust_nav=None, trust_signal=None):
    """Overall trust = the LOWER of the available module scores (either module can raise the alarm)."""
    parts = [t for t in (trust_nav, trust_signal) if t is not None]
    return np.minimum.reduce(parts)
