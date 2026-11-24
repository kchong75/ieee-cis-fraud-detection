# Adapted from: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600 (Chris Deotte)
"""Feature engineering of Chris Deotte's "XGB Fraud with Magic" notebook (1st place team).

Re-implemented on one combined train+test frame instead of two frames mutated in parallel.
Every encoding below uses train *and* test values but never labels (frequency counts,
group means of features), which is the same transductive trick the original uses.

Two feature sets:
    FEATURES_95: base features + frequency/combined/group encodings  (LB ~0.95)
    FEATURES_96: FEATURES_95 + "magic" UID aggregations              (LB ~0.96)

The magic: a UID approximates one client (card):
    uid = card1 + addr1 + floor(day - D1)
D1 is "days since the card was first used", so day - D1 is the card's first-use date and
stays constant for every transaction of that card. We never feed uid to the model. Instead
we aggregate features per uid (mean amount, std of D columns, number of distinct emails...),
so a tree can see "this transaction looks like the other transactions of the same client".
Fraud happens per client, so this helps a lot.
"""
import datetime

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400

# Columns kept by Deotte. 219 of the 339 V columns were dropped as redundant (correlation EDA:
# https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id).
BASE_COLS = (
    ["TransactionAmt", "ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"M{i}" for i in range(1, 10)]
)
V_KEEP = [
    1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47, 48,
    54, 56, 59, 62, 65, 67, 68, 70, 76, 78, 80, 82, 86, 88, 89, 91,
    107, 108, 111, 115, 117, 120, 121, 123, 124, 127, 129, 130, 136,
    138, 139, 142, 147, 156, 162, 165, 160, 166, 178, 176, 173, 182,
    187, 203, 205, 207, 215, 169, 171, 175, 180, 185, 188, 198, 210, 209,
    218, 223, 224, 226, 228, 229, 235, 240, 258, 257, 253, 252, 260, 261,
    264, 266, 267, 274, 277, 220, 221, 234, 238, 250, 271,
    294, 284, 285, 286, 291, 297, 303, 305, 307, 309, 310, 320, 281, 283, 289, 296, 301, 314,
]
ID_COLS = [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"]

# Dropped from the model (but some are still used to build other features):
#  - D6-D9, D12-D14: mostly NaN
#  - the rest failed Deotte's "time consistency" test: a one-feature model trained on the
#    first month scores AUC < 0.5 on the last month, i.e. the feature's meaning drifts.
DROP = (
    ["D6", "D7", "D8", "D9", "D12", "D13", "D14"]
    + ["C3", "M5", "id_08", "id_33", "card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34"]
    + [f"id_{i}" for i in range(22, 28)]
)
NOT_FEATURES = ["TransactionID", "TransactionDT", "isFraud", "is_test", "day", "DT_M", "uid"]


# ---------------------------------------------------------------- encoding helpers
def label_encode(s: pd.Series) -> pd.Series:
    """Map each distinct value to an integer (sorted); NaN -> -1."""
    codes, _ = pd.factorize(s, sort=True)
    return pd.Series(codes, index=s.index, dtype=np.int32)


def freq_encode(df, cols):
    """col_FE = share of rows (train+test) with the same value. Rare values look different."""
    for col in cols:
        vc = df[col].value_counts(normalize=True).to_dict()
        vc[-1] = -1  # -1 means "missing": keep it as -1 rather than its frequency
        df[f"{col}_FE"] = df[col].map(vc).astype(np.float32)


def combine(df, col1, col2):
    """New categorical col1_col2, label-encoded (e.g. card1 within addr1)."""
    name = f"{col1}_{col2}"
    df[name] = label_encode(df[col1].astype(str) + "_" + df[col2].astype(str))
    return name


def group_agg(df, main_cols, group_cols, aggs, usena=True):
    """main_group_agg = statistic of main within each group (e.g. mean amount per card).

    usena=True treats -1 (our "missing" code) as NaN so it doesn't distort mean/std.
    Groups with no value get -1.
    """
    for main in main_cols:
        vals = df[main].where(df[main] != -1) if usena else df[main]
        for grp in group_cols:
            g = vals.groupby(df[grp])
            for agg in aggs:
                df[f"{main}_{grp}_{agg}"] = (
                    g.transform(agg).fillna(-1).astype(np.float32)
                )


def group_nunique(df, main_cols, group_col):
    """group_main_ct = number of distinct values of main within the group."""
    for main in main_cols:
        df[f"{group_col}_{main}_ct"] = (
            df.groupby(group_col)[main].transform("nunique").astype(np.float32)
        )


# ---------------------------------------------------------------- pipeline
def _prepare(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    keep = ["TransactionID", "TransactionDT"] + BASE_COLS + [f"V{v}" for v in V_KEEP] + ID_COLS
    df = pd.concat(
        [train[keep + ["isFraud"]].assign(is_test=False), test[keep].assign(is_test=True)],
        ignore_index=True,
    )

    # Normalize D columns. D_i counts days since some past event, so it grows with time.
    # D_i - day turns it into (minus) the date of that event, which is constant per client
    # and doesn't drift between train and test. D1, D2, D3, D5, D9 are left as they are.
    day_float = df["TransactionDT"] / np.float32(SECONDS_PER_DAY)
    for i in (4, 6, 7, 8, 10, 11, 12, 13, 14, 15):
        df[f"D{i}"] = (df[f"D{i}"] - day_float).astype(np.float32)

    # Label-encode strings. Shift every other numeric column so its min is 0, then NaN -> -1,
    # so XGBoost can treat -1 as "missing" (missing=-1).
    for col in df.columns:
        if col in ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "is_test"):
            continue
        if isinstance(df[col].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = label_encode(df[col])
        else:
            df[col] = (df[col] - df[col].min()).fillna(-1).astype(np.float32)
    return df.copy()


def _base_features(df):
    df["cents"] = (df["TransactionAmt"] - np.floor(df["TransactionAmt"])).astype(np.float32)
    freq_encode(df, ["addr1", "card1", "card2", "card3", "P_emaildomain"])
    combine(df, "card1", "addr1")
    combine(df, "card1_addr1", "P_emaildomain")
    freq_encode(df, ["card1_addr1", "card1_addr1_P_emaildomain"])
    group_agg(df, ["TransactionAmt", "D9", "D11"],
              ["card1", "card1_addr1", "card1_addr1_P_emaildomain"], ["mean", "std"])


def _magic_features(df):
    # Month index (Dec 2017 = 12 ...), as in the original; used in one nunique feature.
    start = pd.Timestamp("2017-11-30")
    dt = start + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["DT_M"] = ((dt.dt.year - 2017) * 12 + dt.dt.month).astype(np.int32)

    # uid = card1_addr1 + first-use date of the card. Note D1 here is already shifted and
    # has NaN as -1, exactly like the original notebook.
    day_float = df["TransactionDT"] / SECONDS_PER_DAY
    df["uid"] = df["card1_addr1"].astype(str) + "_" + np.floor(day_float - df["D1"]).astype(str)

    freq_encode(df, ["uid"])
    group_agg(df, ["TransactionAmt", "D4", "D9", "D10", "D15"], ["uid"], ["mean", "std"])
    group_agg(df, [f"C{i}" for i in range(1, 15) if i != 3], ["uid"], ["mean"])
    group_agg(df, [f"M{i}" for i in range(1, 10)], ["uid"], ["mean"])
    group_nunique(df, ["P_emaildomain", "dist1", "DT_M", "id_02", "cents"], "uid")
    group_agg(df, ["C14"], ["uid"], ["std"])
    group_nunique(df, ["C13", "V314"], "uid")
    group_nunique(df, ["V127", "V136", "V309", "V307", "V320"], "uid")
    # D1 vs D15 disagree by more than 3 days -> transaction may not belong to this client.
    df["outsider15"] = (np.abs(df["D1"] - df["D15"]) > 3).astype(np.int8)


def _feature_list(df):
    return [c for c in df.columns if c not in NOT_FEATURES and c not in DROP]


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Return (df, features_95, features_96). df holds train rows then test rows."""
    df = _prepare(train, test)
    _base_features(df)
    df = df.copy()
    features_95 = _feature_list(df)
    _magic_features(df)
    df = df.copy()
    features_96 = _feature_list(df)
    return df, features_95, features_96
