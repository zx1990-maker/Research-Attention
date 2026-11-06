# PCA baselines for Attention Factors for Statistical Arbitrage

This directory contains a reproducible implementation and out-of-sample results for the paper's two non-attention benchmarks: PCA + OU Threshold and PCA + LongConv. The eight factor counts are 1, 3, 5, 8, 10, 15, 30, and 100. Evaluation covers 6,040 trading days in 1998–2021.

- [Chinese results report](report.md) and [full metrics CSV](results/metrics.csv)
- [Daily and yearly results](results/) for every model and factor count
- [Implementation](run.py), [PCA and accounting](pipeline.py), [LongConv](models.py), and [data audit](audit_data.py)
- [Data audit](audit.json), [method notes](notes/paper_spec.md), and [verification record](results/verification.json)

At K=30, the reproduced gross/net annualized Sharpe ratios are **3.23 / 0.83** for PCA + LongConv and **1.38 / −3.82** for PCA + OU. The paper reports 2.79 / 1.57 and 0.18 / −6.45 respectively. These are transparent method reproductions, not exact author-code replications; see [the report](report.md) for implementation differences.

## Run from source

Use Python 3.12 and install `requirements.txt`. Place `daily_panel.parquet`, `features.parquet`, and `universe_monthly.parquet` in `data/`, or change their paths in `config.json`. Original parquet inputs and the multi-gigabyte PCA cache are not published here. The source files were in a temporary directory that was cleared after audit and PCA preparation. `audit.json` preserves their SHA-256 hashes and the original local result files were verified against saved stock weights before this export.

```bash
python3 -m pip install -r requirements.txt
python3 audit_data.py
python3 run.py prepare
python3 run.py ou --ks 1 3 5 8 10 15 30 100
python3 run.py longconv --ks 1 3 5 8 10 15 30 100
python3 run.py collect
python3 verify_results.py
python3 -m unittest discover -s tests -v
```

`run.py prepare` computes strictly lagged 252-day PCA and writes a SHA-256 sealed local cache. When that cache already exists, `--use-cache` verifies it and runs without reading the parquet inputs. LongConv uses 30 prior daily residual returns, scaled from decimals to percentage points; this unit choice was fixed using 1996–1997 validation before the out-of-sample run. Daily stock weights are normalized to unit gross exposure, with 5 bp L1 turnover and 1 bp daily short costs. The repository tracks summary and daily results; per-stock weight archives and model checkpoints remain in the local workspace because they are substantially larger.
