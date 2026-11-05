import numpy as np

SPOOF_PROFILES = ("step", "linear_drift", "smooth_drift", "oscillating_drift")


def inject(df, n_attacks=None, seed=0, strength=1.0, kinds=("jump", "drift", "jam")):
    """Adds simulated attacks. Works for any drive length.
    jump  = sudden position offset      (spoofing)
    drift = slowly growing offset       (spoofing)
    jam   = frozen position + satellites lost + worse accuracy (jamming)
    'label' = 1 if attacked; 'attack' = 0 normal, 1 spoofing, 2 jamming. strength scales jump/drift size."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    n_attacks = n_attacks or max(3, n // 100)
    slot = n // n_attacks                            # one attack inside each time slot, so train/val/test all get attacks
    Lmax = max(12, min(100, int(slot * 0.6)))
    Lmin = max(5, min(30, Lmax // 3))
    df["label"] = 0
    df["attack"] = 0
    lat, lon = df["lat"].to_numpy().copy(), df["lon"].to_numpy().copy()
    sats, acc = df["numsats"].to_numpy().copy(), df["pos_accuracy"].to_numpy().copy()
    for k in range(n_attacks):
        L = rng.integers(Lmin, Lmax)
        s = max(10, min(k * slot + rng.integers(0, max(1, slot - L - 1)), n - L - 1))
        e, kind = s + L, rng.choice(kinds)
        if kind == "jump":
            lat[s:e] += 0.001 * strength; lon[s:e] += 0.001 * strength
        elif kind == "drift":
            lat[s:e] += np.linspace(0, 0.0008 * strength, L); lon[s:e] += np.linspace(0, 0.0008 * strength, L)
        else:
            lat[s:e] = lat[s]; lon[s:e] = lon[s]
            sats[s:e] = rng.integers(0, 4, L); acc[s:e] = acc[s:e] * 10
        df.loc[df.index[s:e], "label"] = 1
        df.loc[df.index[s:e], "attack"] = 2 if kind == "jam" else 1
    df["lat"], df["lon"], df["numsats"], df["pos_accuracy"] = lat, lon, sats, acc
    return df


def inject_balanced(df, seed=0, spoof_strengths=(1.0,)):
    """Inject balanced spoofing and jamming events into one time segment."""
    if len(df) < 12:
        raise ValueError("A time segment must contain at least 12 rows for balanced attack injection.")

    spoof_strengths = np.asarray(spoof_strengths, dtype=float)
    if spoof_strengths.size == 0 or not np.isfinite(spoof_strengths).all() or (spoof_strengths <= 0).any():
        raise ValueError("spoof_strengths must contain finite positive values.")

    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    n = len(df)
    events_per_type = min(3, max(1, n // 60))
    event_types = [kind for _ in range(events_per_type) for kind in (1, 2)]
    event_count = len(event_types)
    slot_width = n / (event_count + 1)

    df["label"] = 0
    df["attack"] = 0
    df["scenario"] = "normal"
    df["spoof_strength"] = 0.0
    lat = df["lat"].to_numpy(copy=True)
    lon = df["lon"].to_numpy(copy=True)
    sats = df["numsats"].to_numpy(copy=True)
    acc = df["pos_accuracy"].to_numpy(copy=True)
    spoof_index = seed % len(SPOOF_PROFILES)

    for event_index, kind in enumerate(event_types, start=1):
        slot_start = round(event_index * n / (event_count + 1))
        slot_end = round((event_index + 1) * n / (event_count + 1))
        slot_size = slot_end - slot_start
        duration = min(12, max(3, int(slot_size * 0.4) + int(rng.integers(0, 3))))
        recovery = min(3, max(1, duration // 4)) if kind == 1 else 0
        duration = min(duration, slot_size - recovery)
        if duration < 3:
            raise ValueError("Time segment is too short to place non-overlapping attack events.")
        start = slot_start + (slot_size - duration - recovery) // 2
        end = start + duration
        event_end = end + recovery
        df.loc[start:event_end - 1, "label"] = 1
        df.loc[start:event_end - 1, "attack"] = kind

        if kind == 2:
            lat[start:end] = lat[start]
            lon[start:end] = lon[start]
            sats[start:end] = rng.integers(0, 4, end - start)
            acc[start:end] = acc[start:end] * 10
            df.loc[start:end - 1, "scenario"] = "jamming"
            continue

        profile = SPOOF_PROFILES[spoof_index % len(SPOOF_PROFILES)]
        strength = spoof_strengths[spoof_index % len(spoof_strengths)]
        spoof_index += 1
        df.loc[start:event_end - 1, "spoof_strength"] = strength
        t = np.linspace(0, 1, end - start)
        if profile == "step":
            progress = np.ones_like(t)
            magnitude = rng.uniform(20, 120) * strength
        elif profile == "linear_drift":
            progress = t
            magnitude = rng.uniform(10, 80) * strength
        elif profile == "smooth_drift":
            progress = t * t * (3 - 2 * t)
            magnitude = rng.uniform(10, 100) * strength
        else:
            progress = 0.5 - 0.5 * np.cos(2 * np.pi * t)
            magnitude = rng.uniform(10, 50) * strength

        heading = rng.uniform(0, 2 * np.pi)
        full_progress = np.concatenate((progress, np.linspace(progress[-1], 0, recovery + 1)[1:]))
        north = magnitude * np.cos(heading) * full_progress
        east = magnitude * np.sin(heading) * full_progress
        lat[start:event_end] += north / 111320
        cos_lat = np.cos(np.radians(df["lat"].to_numpy()[start:event_end]))
        lon[start:event_end] += east / (111320 * cos_lat)
        df.loc[start:event_end - 1, "scenario"] = profile

    df["lat"], df["lon"], df["numsats"], df["pos_accuracy"] = lat, lon, sats, acc
    return df