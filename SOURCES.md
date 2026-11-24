# Sources

External code and ideas used in this project. Files adapted from a source start with
`# Adapted from: <URL>`. External repos are cloned read-only into `third_party/` (git-ignored).

| Source | Used in | What we took | License |
|---|---|---|---|
| [Chris Deotte — XGB Fraud with Magic 0.9600](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600) | step 1: `src/features.py`, `src/train_baseline.py`, `notebooks/01_baseline.ipynb` | feature engineering, UID aggregations, XGB params (copy in `third_party/deotte_xgb_magic/`) | Kaggle notebook — check before reuse |
| [aws-samples/rgcn-fraud-detector](https://github.com/aws-samples/rgcn-fraud-detector) | step 4 (planned) | RGCN model and graph construction | see its LICENSE |
