import pandas as pd
from src.config import DATA_FILE, ADDRESS_COL, FEATURE_COLUMNS


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    required = [ADDRESS_COL, *FEATURE_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)