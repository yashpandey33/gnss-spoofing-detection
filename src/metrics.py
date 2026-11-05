import numpy as np

def detection_delay(truth, pred, dt=0.1):
    """Average seconds from attack start to first alert (None if no alerts)."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    starts = np.where((truth[1:] > 0) & (truth[:-1] == 0))[0] + 1
    delays = [np.argmax(pred[s:s + 100] > 0) * dt for s in starts if (pred[s:s + 100] > 0).any()]
    return float(np.mean(delays)) if delays else None


def event_recall(truth, pred):
    """(attacks with at least one alert during them, total attacks). Fairer than row-level recall,
    because a constant position offset is only visible to GPS-IMU checks when it starts."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    edges = np.diff(np.r_[0, truth, 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return int(sum(pred[s:e].any() for s, e in zip(starts, ends))), len(starts)