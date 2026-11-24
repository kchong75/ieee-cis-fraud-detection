# Step 1 — Tabular baseline (Deotte's XGB with magic)

Date: 2026-11-23 · Code: `src/features.py`, `src/train_baseline.py` · Notebook: `notebooks/01_baseline.ipynb`
Source: [XGB Fraud with Magic 0.9600](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600) (downloaded to `third_party/`)

## Results (validation AUC)

| Model | Features | Deotte split (75/25) | Deotte reported | **Our split** (day <148 → ≥155) |
|---|---|---|---|---|
| XGB95 (no magic) | 216 | 0.9362 | 0.9363 | **0.9375** |
| XGB96 (magic UID) | 263 | 0.9482 | 0.9472 | **0.9485** |
| Magic gain | +47 | +0.0121 | +0.0109 | **+0.0110** |

**The number to beat in later steps: XGB96 on our split, AUC 0.9485.**

- Reproduced: XGB95 matches to within 0.0001. XGB96 is 0.001 higher than reported, probably
  because of XGBoost version, CPU `hist` vs GPU, and random seed.
- Training on CPU (8 cores, XGBoost 3.4.2): 2–4 min per model; early stopping at around 470–550 trees.
- Both splits agree on the size of the magic gain (about +0.011).
- The validation set is also the early-stopping set, as in the original, so absolute AUCs are
  slightly optimistic. Comparisons between models are fair.

## What the model relies on (XGB96, our split, gain)

- The top features are original Vesta columns: `V258`, `V257`, `V294`, `V187`, `C4`, `V70`.
- The 47 magic features carry **13% of total gain**. The best one is `uid_V309_ct` (rank 21), then
  `uid_C13_ct` and `uid_DT_M_ct` (number of distinct values / months per client).
- So the magic doesn't replace the strong raw features. It adds a "client context" layer
  that is worth about +0.011 AUC.

## Notes for step 2/3

- `uid` has 387k distinct values over train+test vs 55k for `card1_addr1`: it splits cards
  into much finer client groups. Step 2 checks whether graph components recover these groups.
- All encodings use train+test feature values (no labels). This is the same transductive setup
  the graph in step 2 will use.
- Saved: val predictions `data/processed/val_pred_{95,96}_{deotte,ours}.parquet`,
  importances `results/01_importance_*.csv`, raw run log `results/01_train.log`.
