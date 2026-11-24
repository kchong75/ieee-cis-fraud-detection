# Adapted from: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600 (Chris Deotte)
# Original notebook released under the Apache License 2.0. Modified by kchong75 (2026):
# restructured into reusable modules, run on a time-based validation split.
"""Step 1: train Deotte's XGB95 (no magic) and XGB96 (magic UID features) on two splits.

Splits:
  deotte  - first 75% of train rows -> last 25%  (to check we reproduce 0.9363 / 0.9472)
  ours    - src/validation.py: day < 148 -> day >= 155  (the split all later steps use)

Like the original, the validation set is also the early-stopping set, which makes the AUC
slightly optimistic. Fine for comparing feature sets, since every model gets the same treatment.

Usage:  python -m src.train_baseline            # all 4 runs
        python -m src.train_baseline ours 96    # one run
Writes results/01_baseline_runs.json, results/01_importance_<model>_<split>.csv and
data/processed/val_pred_<model>_<split>.parquet.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.data import PROCESSED, ROOT, load
from src.features import build_features
from src.validation import time_split

RESULTS = ROOT / "results"
RUNS_FILE = RESULTS / "01_baseline_runs.json"

# Deotte's parameters; only the GPU tree method is swapped for CPU "hist".
XGB_PARAMS = dict(
    n_estimators=2000,
    max_depth=12,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.4,
    missing=-1,
    eval_metric="auc",
    early_stopping_rounds=100,
    tree_method="hist",
    n_jobs=8,
    random_state=42,
)


def split_masks(train_part: pd.DataFrame, name: str):
    if name == "deotte":
        n = len(train_part)
        idx = np.arange(n)  # rows are already sorted by TransactionDT
        return idx < 3 * n // 4, idx >= 3 * n // 4
    if name == "ours":
        tr, va = time_split(train_part)
        return tr.to_numpy(), va.to_numpy()
    raise ValueError(name)


def run(df, features, model_name, split_name, raw_train):
    train_part = df[~df["is_test"]]
    tr, va = split_masks(raw_train, split_name)
    X_tr, y_tr = train_part.loc[tr, features], train_part.loc[tr, "isFraud"]
    X_va, y_va = train_part.loc[va, features], train_part.loc[va, "isFraud"]
    print(f"\n== XGB{model_name} on split '{split_name}': {len(features)} features, "
          f"train {len(X_tr):,} rows, val {len(X_va):,} rows", flush=True)

    t = time.time()
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
    minutes = (time.time() - t) / 60

    pred = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    print(f"   val AUC = {auc:.4f}  (best iteration {clf.best_iteration}, {minutes:.1f} min)", flush=True)

    tag = f"{model_name}_{split_name}"
    gain = clf.get_booster().get_score(importance_type="gain")
    (pd.Series(gain, name="gain").reindex(features).fillna(0).sort_values(ascending=False)
       .to_csv(RESULTS / f"01_importance_{tag}.csv", index_label="feature"))
    pd.DataFrame({"TransactionID": train_part.loc[va, "TransactionID"].to_numpy(),
                  "isFraud": y_va.to_numpy(), "pred": pred}
                 ).to_parquet(PROCESSED / f"val_pred_{tag}.parquet", index=False)
    return dict(model=f"XGB{model_name}", split=split_name, n_features=len(features),
                train_rows=int(tr.sum()), val_rows=int(va.sum()), val_auc=round(auc, 5),
                best_iteration=int(clf.best_iteration), train_minutes=round(minutes, 1),
                xgboost=xgb.__version__)


def main(splits=("deotte", "ours"), models=("95", "96")):
    RESULTS.mkdir(exist_ok=True)
    raw_train, raw_test = load("train"), load("test")
    df, f95, f96 = build_features(raw_train, raw_test)
    feats = {"95": f95, "96": f96}
    runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else []
    for split in splits:
        for model in models:
            res = run(df, feats[model], model, split, raw_train)
            runs = [r for r in runs if not (r["model"] == res["model"] and r["split"] == split)]
            runs.append(res)
            RUNS_FILE.write_text(json.dumps(runs, indent=2))  # save after every run
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*( [(args[0],), (args[1],)] if len(args) == 2 else [] ))
