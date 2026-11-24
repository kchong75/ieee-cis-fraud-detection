# IEEE-CIS Fraud Detection — learning graph analysis

A personal learning project: graph analysis for fraud detection on the Kaggle
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset
(Vesta e-commerce transactions, target `isFraud`, metric ROC-AUC).
The goal is to understand the methods, not to top the leaderboard.

## Plan and status

| Step | What | Status |
|---|---|---|
| 0 | Setup, Parquet cache, time-based validation split | done |
| 1 | Tabular baseline: Chris Deotte's XGB with "magic" UID features | done, val AUC **0.9485** |
| 2 | Transaction–entity graph: degrees, connected components, UID recovery | in progress |
| 3 | Graph-derived features added to the GBDT | planned |
| 4 | RGCN on the heterogeneous graph | planned |
| 5 | Write-up | planned |

Details: [plan.md](plan.md). Results per step: [results/](results/).

## Layout

```
src/        loading, features, graph building, training
notebooks/  one notebook per step (01_baseline, 02_graph_eda, ...)
results/    metrics and short notes per experiment
```

## Running it

1. Download the competition data from Kaggle and unzip it into `data/` inside this folder (`proj2_graph_analysis/data/`).
   Run all commands below from this folder.
   **The data is not in this repo**: the competition rules don't allow redistributing it.
2. Build the Parquet cache: `python -m src.data`
3. Train the baseline: `python -m src.train_baseline`, or open the notebooks.

Python 3.11 with pandas, numpy, scikit-learn, xgboost, scipy, pyarrow.

## Credits

- **Chris Deotte**, [XGB Fraud with Magic 0.9600](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600).
  The step 1 feature engineering and model setup are adapted from this notebook (Apache 2.0).
- **FraudSquad (1st place)**, [write-up](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284):
  the "predict clients, not transactions" idea and the UID construction.
- **CPMP (2nd place)**, [write-up](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/2-uncles-and-3-puppies-2nd-solution-cpmp-view).
- **AWS**, [aws-samples/rgcn-fraud-detector](https://github.com/aws-samples/rgcn-fraud-detector):
  the reference for the planned RGCN step.

See [SOURCES.md](SOURCES.md) for exactly which files use which source.

## License

[Apache License 2.0](../LICENSE). Adapted files keep their original attribution.
