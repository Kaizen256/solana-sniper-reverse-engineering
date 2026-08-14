# Solana Sniper Bot Reverse-Engineering

Reconstruction of the Kaggle target wallet `5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`.

The pipeline enforces a strict deployment-time decision boundary. Current-token features come only from information available at deployment. Historical deployer state and historical target-wallet behavior may contribute only when they were observed strictly before the candidate deployment. In particular, a prior target buy enters the relationship state only when:

`target_buy_time < current candidate block_time`

Equality, the target reaction to the current token, and all future actions are excluded. Transaction execution metadata, June trades, candles, and landed Jito information are evaluation-only.

## Results at a glance

- 15,927 core bot-bought deployment tokens; 79.63% were bought in the deployment slot.
- Median same-slot position: +118 transactions after deployment.
- Median hold: 6 seconds; 96.59% of bought positions use partial exits.
- Historical target-wallet cash flow: **+$925,056 fully fee-adjusted P&L**, with a **58.65% hit rate**.
- Promoted target-signer classifier:
  - **April PR-AUC: 0.282063**
  - **May PR-AUC: 0.385999**
  - **June PR-AUC: 0.2047103771**
- Frozen June operating point: **29.32% precision**, **42.60% recall**, **0.34736 F1**, **6,094 selections**, and **1,787 true positives**.
- The dominant feature is `deployments_since_prior_target_buy`, accounting for **58.59% of model gain**.
- Primary marginal backtest:
  - immediate execution: **66.74% hit rate, +20.27% median ROI**
  - +1 slot: **34.93% hit rate, -10.71% median ROI**
- Actual target June activity: **+$185,610 fully fee-adjusted P&L** and **16.79% ROI**.
- Exact Pump +118 replay for the selective strategy:
  - **46.86-47.93% supported coverage**
  - **-8.46% to -7.06% median fully modeled ROI**
  - **-9.0 to +38.2 SOL p99-capped P&L at 0.95% fees**
  - **-17.0 to +29.9 SOL at 1.25% fees**

The exact replay is a bounded secondary analysis. The source-built marginal backtest remains the primary Part 3 result.

## Repository map

```text
scripts/               bootstrap audit, full runner, notebook/cover builder
src/.../behavior.py    target-wallet reconstruction
src/.../message_features.py  one-pass signed-message decoder
src/.../feature_store.py     strict-prior ASOF feature pipeline
src/.../modeling.py    chronological baselines, final model, interpretation
src/.../backtest.py    frozen June strategy and execution sensitivities
src/.../robustness.py  alternate split, stability, calibration, subgroups
src/.../methodological_audit.py  active-era, redundancy, and intra-slot controls
src/.../third_pass.py   outcome quality, drift, ranking, hard negatives, strategy
src/.../curve_replay.py integer Pump curve impact and intent bounds
src/.../raw_block_audit.py  targeted raw event/fee/instruction validation
tests/                 temporal, composite-key, legality, and accounting tests
submission/            notebook, <=3,000-word writeup, figures, small tables
```

Raw and generated large data are ignored by Git. Nothing under `data/raw/` is modified.

## Data layout

Competition data is not included in this repository. If you want to reproduce the results, place the files at these paths without renaming their contents:

```text
data/raw/core/bought_deploy_txs.jsonl.gz
data/raw/core/bought_deploy_txs_index.parquet
data/raw/core/bought_deployers_activity.parquet
data/raw/core/not_bought_deploy_txs.jsonl.gz
data/raw/core/not_bought_deploy_txs_index.parquet
data/raw/core/not_bought_deployers_activity.parquet
data/raw/target_wallet/5brv79e_activity.parquet
data/raw/target_wallet/5brv79e_activity_txs.jsonl.gz
data/raw/target_wallet/5brv79e_activity_txs_index.parquet
data/raw/june/trades/pumpfun_trades.parquet
data/raw/june/candles/mcap_candles.parquet
data/raw/june/jito/jito_bundle_transactions.parquet
```

The core deployment archives, deployer-activity tables, and target-wallet files drive
behavior reconstruction and strict-prior feature construction. June trades and
candles are post-deployment evaluation data only; the Jito transaction map is used
only for execution analysis. None of these may become entry features. Raw files are
immutable inputs, and generated caches belong only under `data/interim/`,
`data/processed/`, or `artifacts/`.

## Reproduce

Python 3.12 and `uv` are expected.

```bash
UV_CACHE_DIR=/tmp/solana-main-uv-cache uv sync
.venv/bin/pytest -q
MPLCONFIGDIR=/tmp/solana-mpl .venv/bin/python scripts/run_pipeline.py
MPLCONFIGDIR=/tmp/solana-mpl .venv/bin/python scripts/run_third_pass.py all
```

The notebook provides the judge-facing end-to-end orchestration path over the same tested pipeline:

```bash
MPLCONFIGDIR=/tmp/solana-mpl .venv/bin/python scripts/execute_notebook_cells.py submission/final_notebook.ipynb
```

After either path, rebuild and audit the small submission package:

```bash
MPLCONFIGDIR=/tmp/solana-mpl .venv/bin/python scripts/build_submission.py
MPLCONFIGDIR=/tmp/solana-mpl .venv/bin/python scripts/reproducibility_audit.py --isolated-positive-extraction
```

The first clean run streams the 15 GiB negative gzip once, groups 177.4M activity rows into strict-prior states, fits chronological models, and scans only needed June outcome rows. On the reference 24-core/60 GiB host, signed-message extraction took about 8 minutes and the feature store about 4 minutes. DuckDB may spill roughly 50 GiB while building the activity state; ensure adequate workspace storage. Subsequent deterministic-cache runs are much faster.

Individual stages are also runnable:

```bash
.venv/bin/python -m solana_sniper_reverse_engineering.behavior
.venv/bin/python -m solana_sniper_reverse_engineering.message_features
.venv/bin/python -m solana_sniper_reverse_engineering.feature_store
.venv/bin/python -m solana_sniper_reverse_engineering.modeling
.venv/bin/python -m solana_sniper_reverse_engineering.backtest --force
.venv/bin/python -m solana_sniper_reverse_engineering.robustness
.venv/bin/python -m solana_sniper_reverse_engineering.methodological_audit
.venv/bin/python scripts/run_third_pass.py historical
.venv/bin/python scripts/run_curve_replay.py
```

Large deterministic outputs and models live in ignored `data/interim/`, `data/processed/`, and `artifacts/`. Small judge-facing results are in `submission/tables/`; every number in the writeup maps to one of these files. `final_selector_feature_importance.csv` and `final_selector_feature_effects.csv` are deterministic presentation extracts from the frozen promoted selector, not another fit or model-selection pass.

`reproducibility_audit.py` validates raw-source identities against manifests, full feature temporal/key gates, cross-table claims, notebook paths, Git publication hygiene, and a clean isolated extraction of the positive raw gzip. The notebook contains plain Python cells, so `execute_notebook_cells.py` provides a local execution gate even when Jupyter/`nbconvert` is not installed.

## Submission package

- `submission/final_notebook.ipynb`
- `submission/writeup.md`
- `submission/figures/cover.png` plus seven analysis figures
- `submission/tables/` traceable summaries, feature dictionary, effects, and robustness tables

## License

The source code in this repository is licensed under the MIT License.

Competition data is not included in this repository and is not covered by
the MIT License. It remains subject to the competition rules and the data
provider's terms.