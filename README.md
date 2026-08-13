# Solana Sniper Bot Reverse-Engineering

Competition-ready reconstruction and replica of the Kaggle target wallet `5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`.

The pipeline preserves one hard boundary: entry features may use only the signed deployment transaction and deployer facts already observed strictly before deployment. A prior launch outcome enters only after its timestamped creator-fee claim or deployer sell; future outcomes never backfill history. Target reactions, future history, transaction execution metadata, landed Jito data, June trades, and candles are never model inputs.

## Results at a glance

- 15,927 core bot-bought tokens; 79.63% bought in the deployment slot.
- Median hold: 6 seconds; 96.59% use partial exits.
- Active-era May validation: PR-AUC 0.1445, precision 21.85%, recall 26.41%, F1 0.2392.
- June reporting test: PR-AUC 0.0687, precision 14.45%, recall 13.44%, F1 0.1393. PR-AUC is 13.96× prevalence; precision is 29.34× prevalence.
- Strict-as-of creator-fee plus developer-sell history improves paired April/May PR-AUC from 0.09705/0.14465 to 0.10756/0.15337. Its June PR-AUC is 0.06542, reported without tuning.
- The controlled January–April model is weaker on May and June PR-AUC, so the final trains from the target's 2026-03-12 active-era start.
- A separate economic ranker improves seven-day creator-claim hit rate in both pre-June windows. Its selective June strategy makes 1,121 entries.
- At +118 positions, integer Pump replay gives the selective strategy −7.14% to −6.34% median ROI but +9.8 to +51.1 SOL p99-capped P&L on 50.58–51.56% supported coverage. See the writeup for bounds.

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

Competition data is not included in this repository. If you wish to replicate,
obtain the authorized files, and place them at these paths without renaming their
contents:

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

The optional roughly 429 GiB raw-block supplement is not required. This project used
one targeted batch to validate event decoding and pricing mechanics; reproducing the
main notebook does not require downloading the full supplement.

## Reproduce

Python 3.12 and `uv` are expected.

```bash
UV_CACHE_DIR=/tmp/solana-main-uv-cache uv sync
.venv/bin/pytest -q
MPLCONFIGDIR=/tmp/solana-mpl .venv/bin/python scripts/run_pipeline.py
MPLCONFIGDIR=/tmp/solana-mpl .venv/bin/python scripts/run_third_pass.py all
```

The notebook is an alternative end-to-end orchestration path, not an additional required rerun:

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
