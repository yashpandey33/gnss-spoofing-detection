import glob, os, numpy as np, pandas as pd


def _candidate_kitti_dirs(folder=None):
    """Search a few common places for a KITTI oxts/data folder."""
    roots = []
    if folder:
        roots.append(os.path.abspath(folder))
    cwd = os.path.abspath(os.getcwd())
    roots.extend([
        cwd,
        os.path.dirname(cwd),
        os.path.dirname(os.path.dirname(cwd)),
    ])
    seen = set()
    for root in roots:
        for name in ("oxts", "oxts2", "oxts3"):
            candidate = os.path.join(root, name, "data")
            if candidate not in seen and os.path.isdir(candidate):
                seen.add(candidate)
                yield candidate
    # Also accept an already-resolved dataset directory even when the path is spelled with a file name.
    if folder and os.path.isdir(folder):
        yield os.path.abspath(folder)


def discover_kitti_dirs(root=None):
    """Find KITTI oxts/data directories alongside the project or working directory."""
    roots = []
    if root:
        root = os.path.abspath(root)
        roots.extend((root, os.path.dirname(root), os.path.dirname(os.path.dirname(root))))
    cwd = os.path.abspath(os.getcwd())
    roots.extend((cwd, os.path.dirname(cwd), os.path.dirname(os.path.dirname(cwd))))

    found = []
    seen = set()
    for base in roots:
        for candidate in sorted(glob.glob(os.path.join(base, "oxts*", "data"))):
            if candidate not in seen and glob.glob(os.path.join(candidate, "*.txt")):
                found.append(candidate)
                seen.add(candidate)
    return found


def _resolve_kitti_dir(folder=None):
    if folder is None:
        for candidate in _candidate_kitti_dirs():
            if glob.glob(os.path.join(candidate, "*.txt")):
                return candidate
        return None

    folder = os.path.abspath(folder)
    if os.path.isdir(folder) and glob.glob(os.path.join(folder, "*.txt")):
        return folder

    for candidate in _candidate_kitti_dirs(folder):
        if glob.glob(os.path.join(candidate, "*.txt")):
            return candidate
    return None


def load_kitti(folder=None):
    """folder = ...\\2011_09_26_drive_00XX_sync\\oxts\\data  (10 Hz)"""
    if folder is None:
        folder = _resolve_kitti_dir()
        if folder is None:
            return synthetic()

    folder = _resolve_kitti_dir(folder)
    if folder is None:
        raise SystemExit(
            f"No .txt files in {folder if folder else 'the requested KITTI folder'}. "
            "Use a real oxts\\data folder or run from a project directory that contains one."
        )

    files = sorted(glob.glob(f"{folder}/*.txt"))
    a = np.array([np.loadtxt(f) for f in files])
    return pd.DataFrame({"lat": a[:, 0], "lon": a[:, 1], "imu_speed": a[:, 8],      # vf = forward speed
                         "vn": a[:, 6], "ve": a[:, 7],                               # IMU/INS velocity north, east
                         "pos_accuracy": a[:, 23], "navstat": a[:, 25], "numsats": a[:, 26]})


def synthetic(n=6000):
    """Fallback: car driving a wavy path at ~10 m/s (works without KITTI)."""
    rng = np.random.default_rng(1)
    t = np.arange(n) * 0.1
    vn = 7 * (1 + 0.3 * np.sin(t / 20))
    ve = 7 * (1 + 0.3 * np.cos(t / 15))
    lat = 12.97 + np.cumsum(vn * 0.1) / 111320
    lon = 77.59 + np.cumsum(ve * 0.1) / (111320 * np.cos(np.radians(12.97)))
    return pd.DataFrame({"lat": lat, "lon": lon, "imu_speed": np.hypot(vn, ve), "vn": vn, "ve": ve,
                         "pos_accuracy": 0.02 + rng.normal(0, 0.002, n).clip(-0.01, 0.01),
                         "navstat": 4.0, "numsats": rng.integers(9, 13, n).astype(float)})

def load(folder=None):
    return load_kitti(folder)
