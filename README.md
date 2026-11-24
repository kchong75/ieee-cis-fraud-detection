# IEEE-CIS Fraud Detection — graph-based fraud detection projects

Projects on the Kaggle [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
dataset (Vesta e-commerce transactions, binary target `isFraud`). Each folder is a
self-contained project with its own README, code and instructions.

| Project | Question | Status |
|---|---|---|
| [proj1_card_pair_graph_features](proj1_card_pair_graph_features/) | Does recent activity on the same `(card1, card2)` pair help catch more fraud at a fixed review capacity? | done: gains were not consistent across time windows |
| [proj2_graph_analysis](proj2_graph_analysis/) | How far can a transaction–entity graph go, from graph features in a GBDT to an RGCN, against a strong tabular baseline? | in progress: baseline XGB val AUC 0.9485 |

## Data

The data is **not included**: the competition rules don't allow redistributing it.
Download it from the [competition page](https://www.kaggle.com/c/ieee-fraud-detection/data).
Each project's README says where it expects the files.

## License

[Apache License 2.0](LICENSE). Code adapted from other authors keeps its original
attribution (see each project's README or `SOURCES.md`).
