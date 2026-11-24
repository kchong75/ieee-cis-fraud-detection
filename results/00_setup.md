# Step 0 — Setup

Date: 2026-11-23

## Environment

- Miniconda (`~/miniconda3`), env **`env_proj_ieee_cis_fraud_detection`**, Python 3.11.18, native arm64.
- conda-forge: pandas 3.1.1, numpy 2.4.6, pyarrow 26.0.1, scikit-learn 1.9.1, scipy 1.17.1,
  xgboost 3.4.2, lightgbm 4.7.0, networkx 3.6.1, matplotlib 3.11.2, jupyterlab.
- torch / dgl not installed yet (step 4).
- Hardware: 16 GB RAM, 8 cores.

## Data cache (`python -m src.data`)

| Split | Shape | Memory | Parquet | Load time | Days | Has identity |
|---|---|---|---|---|---|---|
| train | 590,540 × 435 | 0.97 GB | 80 MB | 2.9 s | 1–182 | 24.4% |
| test | 506,691 × 434 | 0.83 GB | 70 MB | 0.6 s | 213–395 | 28.0% |

- Types after shrinking: 399 float32, 31 category (text columns), 5 int32.
- Building the cache from CSV takes about 22 s.
- `id-01` → `id_01` rename in test confirmed.
- Test starts about 30 days after train ends (day 213 vs 182), so a gap before validation
  mirrors the real setup.

## Validation split (`src/validation.py`)

| Part | Days | Rows | Share | Fraud rate |
|---|---|---|---|---|
| train | 1–147 | 490,555 | 83.1% | 3.51% |
| gap (unused) | 148–154 | 21,443 | 3.6% | 3.08% |
| val | 155–182 | 78,542 | 13.3% | 3.53% |

Fraud rate is steady across the split, so validation AUC won't be skewed by a shift in how
common fraud is.

## Notes

- pandas 3 reads text as the `str` dtype rather than `object`; `src/data.py` handles both.
