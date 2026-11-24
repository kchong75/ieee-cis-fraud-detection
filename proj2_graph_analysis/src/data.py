"""Load the IEEE-CIS raw CSVs once, shrink them, and cache them as Parquet.

Why: each raw transaction CSV is ~650 MB and pandas' default float64/int64 types make it
1.5-2 GB in memory. Most columns fit in float32 (half the size), and Parquet loads in
seconds instead of the ~1 minute a CSV parse takes.

Usage:
    python -m src.data            # build data/processed/{train,test}.parquet
    from src.data import load     # later: df = load("train")
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
PROCESSED = RAW / "processed"

SECONDS_PER_DAY = 86400


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # test_identity uses "id-01"; train uses "id_01". Normalise to underscores.
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> int32 (all ints in this dataset fit), strings -> category."""
    new_types = {}
    for col, dtype in df.dtypes.items():
        if dtype == np.float64:
            new_types[col] = np.float32
        elif dtype == np.int64:
            new_types[col] = np.int32
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            # pandas 3 reads text as the "str" dtype, older pandas as object; handle both.
            new_types[col] = "category"
    # One astype call builds a single compact frame (per-column assignment fragments it).
    return df.astype(new_types)


def build_raw(split: str) -> pd.DataFrame:
    """Left-join transaction + identity for one split ('train' or 'test')."""
    trans = _downcast(_read_csv(RAW / f"{split}_transaction.csv"))
    ident = _downcast(_read_csv(RAW / f"{split}_identity.csv"))
    # Left join: only ~25% of transactions have identity rows; the rest get NaN.
    # .copy() consolidates the ~430 per-column blocks left by astype, so adding a column is cheap.
    df = trans.merge(ident, on="TransactionID", how="left").copy()
    df["day"] = (df["TransactionDT"] // SECONDS_PER_DAY).astype(np.int32)
    return df


def cache(split: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"{split}.parquet"
    build_raw(split).to_parquet(out, index=False)
    return out


def load(split: str) -> pd.DataFrame:
    """Load the cached Parquet for a split (run `python -m src.data` first)."""
    return pd.read_parquet(PROCESSED / f"{split}.parquet")


if __name__ == "__main__":
    for s in ("train", "test"):
        print("cached", cache(s))
