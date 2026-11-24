"""The single time-based validation split used by every experiment.

Why time-based: Kaggle's test set comes *after* train in time, and test labels are hidden.
To estimate how well a model predicts the future, we hold out the most recent weeks of
train as validation. A random split would put transactions of the same card on both
sides and overstate the AUC.

Why a gap: features such as "days since card first seen" or rolling counts can carry
information across the boundary. A one-week gap that is used for neither training nor
validation makes the estimate more honest.
"""
import pandas as pd

# train spans day 1..182. Validate on the last 4 weeks, skip the week before it.
VAL_START_DAY = 155
GAP_DAYS = 7
TRAIN_END_DAY = VAL_START_DAY - GAP_DAYS  # train on day < 148


def time_split(df: pd.DataFrame):
    """Return boolean masks (train_mask, val_mask) over df using its `day` column."""
    train_mask = df["day"] < TRAIN_END_DAY
    val_mask = df["day"] >= VAL_START_DAY
    return train_mask, val_mask
