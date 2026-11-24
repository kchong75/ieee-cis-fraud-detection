# Card-Pair Activity for Transaction Fraud Detection

Does recent activity on the same card pair help a fraud model catch more fraud at a fixed review capacity? This notebook tests a graph-derived feature on the IEEE-CIS Fraud Detection dataset and measures whether its incremental value holds across time.

## Method

- Represent transactions and `(card1, card2)` pairs as nodes in a bipartite graph. Each transaction links to its card pair.
- For each transaction, count earlier transactions linked to the same pair in the preceding seven days, using the interval `[t − 7 days, t)`. Transactions at the same timestamp are excluded.
- Compare a baseline model with a version that adds this feature across three chronological validation windows. For each window, rank transactions by predicted fraud score and compare fraud capture among the top 590. Also report the change in average precision (AP) across the full ranking.

## Results

| Validation window | Baseline frauds caught | With graph feature | Change in frauds caught | Change in AP |
| --- | ---: | ---: | ---: | ---: |
| 40%–50% | 308 | 334 | +26 | +0.88 pp |
| 50%–60% | 213 | 194 | −19 | −1.36 pp |
| 60%–70% | 198 | 197 | −1 | −1.45 pp |

The window labels are percentile ranges in chronological transaction order. Each model had the same review budget of 590 transactions per window. AP changes are in percentage points.

The feature increased fraud capture in the first window. The second window lost 19 detections, and the third was nearly flat at the review cutoff while AP fell. The baseline remains the better default given the inconsistent gains across windows. A follow-up would examine which transactions moved into and out of the review queue and whether their card-pair activity patterns changed over time.

## Run the notebook

1. Download the data from the [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data).
2. Install the Python dependencies with `pip install -r requirements.txt`.
3. Open `fraud_graph_analysis.ipynb` in Jupyter, set the input path in the data-loading cell to your local copy of the dataset, and run the cells in order.
