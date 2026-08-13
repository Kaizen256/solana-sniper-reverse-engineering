from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
SUBMISSION = ROOT / "submission"

BOUGHT_INDEX = RAW / "core" / "bought_deploy_txs_index.parquet"
NOT_BOUGHT_INDEX = RAW / "core" / "not_bought_deploy_txs_index.parquet"
BOUGHT_ACTIVITY = RAW / "core" / "bought_deployers_activity.parquet"
NOT_BOUGHT_ACTIVITY = RAW / "core" / "not_bought_deployers_activity.parquet"
BOUGHT_TXS = RAW / "core" / "bought_deploy_txs.jsonl.gz"
NOT_BOUGHT_TXS = RAW / "core" / "not_bought_deploy_txs.jsonl.gz"
TARGET_ACTIVITY = RAW / "target_wallet" / "5brv79e_activity.parquet"
TARGET_TXS = RAW / "target_wallet" / "5brv79e_activity_txs.jsonl.gz"
TARGET_TX_INDEX = RAW / "target_wallet" / "5brv79e_activity_txs_index.parquet"
JUNE_TRADES = RAW / "june" / "trades" / "pumpfun_trades.parquet"
JUNE_CANDLES = RAW / "june" / "candles" / "mcap_candles.parquet"
JITO_TRANSACTIONS = RAW / "june" / "jito" / "jito_bundle_transactions.parquet"

MAY_START = 1_777_593_600  # 2026-05-01T00:00:00Z
JUNE_START = 1_780_272_000  # 2026-06-01T00:00:00Z
JULY_START = 1_782_864_000  # 2026-07-01T00:00:00Z
ACTIVE_ERA_START = 1_773_273_600  # 2026-03-12T00:00:00Z


def ensure_output_dirs() -> None:
    for path in (INTERIM, PROCESSED, ARTIFACTS, SUBMISSION):
        path.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "figures").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "tables").mkdir(parents=True, exist_ok=True)
    (SUBMISSION / "figures").mkdir(parents=True, exist_ok=True)
    (SUBMISSION / "tables").mkdir(parents=True, exist_ok=True)
