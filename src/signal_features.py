import pandas as pd

def add_rolling(df, win=5):
    """Adds 5-step rolling mean/std of C/N0 (used by signal_prep and the API)."""
    df = df.copy()
    df["cn0_mean5"] = df["cn0"].rolling(win).mean().bfill()
    df["cn0_std5"] = df["cn0"].rolling(win).std().bfill().fillna(0)
    return df
