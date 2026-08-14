from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from .config import (
    ACTIVE_ERA_START,
    ARTIFACTS,
    BOUGHT_ACTIVITY,
    BOUGHT_INDEX,
    INTERIM,
    JUNE_TRADES,
    JUNE_START,
    MAY_START,
    NOT_BOUGHT_ACTIVITY,
    NOT_BOUGHT_INDEX,
    PROCESSED,
    SUBMISSION,
    ensure_output_dirs,
)
from .backtest import _metrics as backtest_metrics
from .backtest import _strategy_parameters
from .feature_store import ACTIVITY_WALLETS, BASE_FEATURES, FEATURE_STORE
from .modeling import (
    MODEL_DIR,
    ModelBundle,
    _available_features,
    _preprocessor,
    _thresholds,
    fit_lightgbm,
    metrics,
    predict,
)


APRIL_START = 1_775_001_600
EPOCH_2026 = 1_767_225_600
DAY = 86_400
RANKING_BUCKET_SECONDS = 6 * 3_600

QUALITY_LIFECYCLE = INTERIM / "historical_launch_quality_lifecycle.parquet"
QUALITY_STATE = INTERIM / "historical_launch_quality_state.parquet"
QUALITY_FEATURES = INTERIM / "historical_launch_quality_features.parquet"
DEV_SELL_LIFECYCLE = INTERIM / "historical_launch_dev_sell_lifecycle.parquet"
DEV_SELL_STATE = INTERIM / "historical_launch_dev_sell_state.parquet"
DEV_SELL_FEATURES = INTERIM / "historical_launch_dev_sell_features.parquet"
CLAIM_OUTCOMES = INTERIM / "candidate_creator_fee_outcomes.parquet"
METADATA_FEATURES = INTERIM / "third_pass_metadata_features.parquet"

MODEL_RESULTS = SUBMISSION / "tables" / "third_pass_modeling.json"
QUALITY_MODEL_DIR = ARTIFACTS / "models" / "quality_augmented_final"
QUALITY_AUDIT = SUBMISSION / "tables" / "historical_outcome_audit.json"
QUALITY_DAILY_STABILITY = SUBMISSION / "tables" / "historical_outcome_daily_stability.csv"
DEV_SELL_RESULTS = SUBMISSION / "tables" / "developer_sell_outcome_results.json"
POLICY_RESULTS = SUBMISSION / "tables" / "ranking_hard_negative_results.json"
WEEKLY_RESULTS = SUBMISSION / "tables" / "weekly_regime_analysis.json"
WEEKLY_TABLE = SUBMISSION / "tables" / "weekly_regime_metrics.csv"
WEEKLY_EFFECTS = SUBMISSION / "tables" / "weekly_feature_effects.csv"
ECONOMIC_RESULTS = SUBMISSION / "tables" / "profitable_disagreement_results.json"
STRATEGY_FRONTIER = SUBMISSION / "tables" / "pre_june_strategy_frontier.csv"
DISAGREEMENT_COHORTS = SUBMISSION / "tables" / "profitable_disagreement_cohorts.csv"
METADATA_RESULTS = SUBMISSION / "tables" / "signed_message_metadata_results.json"
THIRD_STRATEGY_OUTCOMES = ARTIFACTS / "tables" / "third_pass_strategy_outcomes.parquet"

HISTORICAL_OUTCOME_FEATURES = [
    "quality_history_missing",
    "quality_launch_history_incomplete",
    "hist_pump_launch_count",
    "hist_claimed_launch_count",
    "hist_claim_event_count",
    "hist_claim_fee_sol_sum",
    "hist_claim_fee_usd_sum",
    "hist_claim_fee_usd_max",
    "hist_claimed_launch_fraction",
    "hist_claimed_launch_per_core_deploy",
    "hist_claim_fee_usd_per_claimed_launch",
    "seconds_since_claim_fee",
    "hist_mature_1d_launch_count",
    "hist_mature_1d_success_count",
    "hist_mature_1d_success_fraction",
    "hist_mature_7d_launch_count",
    "hist_mature_7d_success_count",
    "hist_mature_7d_success_fraction",
    "hist_claimed_launch_count_7d",
    "hist_claimed_launch_count_30d",
    "hist_claim_fee_usd_7d",
    "hist_claim_fee_usd_30d",
    "hist_decayed_30d_success_fraction",
    "latest_prior_launch_claimed",
    "latest_prior_launch_mature_1d_failure",
    "latest_prior_launch_mature_7d_failure",
    "latest_prior_launch_claim_latency_seconds",
]

DEVELOPER_SELL_FEATURES = [
    "hist_dev_sold_launch_count",
    "hist_dev_sold_launch_fraction",
    "seconds_since_prior_dev_sell",
    "hist_mature_1d_dev_sold_count",
    "hist_mature_1d_dev_sold_fraction",
    "hist_mature_7d_dev_sold_count",
    "hist_mature_7d_dev_sold_fraction",
    "hist_dev_sold_launch_count_7d",
    "hist_dev_sold_launch_count_30d",
    "latest_prior_launch_group_dev_sold_fraction",
    "latest_prior_launch_group_mature_1d_no_dev_sell_fraction",
    "latest_prior_launch_group_mature_7d_no_dev_sell_fraction",
    "latest_prior_launch_group_dev_sell_latency_median_seconds",
]

METADATA_REUSE_FEATURES = [
    "prior_uri_count",
    "seconds_since_prior_uri",
    "signer_prior_uri_count",
    "prior_uri_host_count",
    "signer_prior_uri_host_count",
    "prior_name_symbol_count",
    "signer_prior_name_symbol_count",
    "priority_fee_intent_lamports",
    "dev_buy_instruction_gap",
    "dev_buy_immediately_after_create",
    "create_instruction_fraction",
    "system_transfer_mean_sol",
]


def _sql(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _connection(memory: str = "36GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    con.execute("SET threads=20")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    temp = INTERIM / "duckdb_temp_third_pass"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{_sql(temp)}'")
    return con


def _canonical_activity_sql(columns: str, predicate: str) -> str:
    """Use the same non-duplicating wallet-source policy as the feature store."""
    return f"""
      SELECT {columns}
      FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')
      WHERE {predicate}
      UNION ALL
      SELECT {', '.join('b.' + item.strip() for item in columns.split(','))}
      FROM read_parquet('{_sql(BOUGHT_ACTIVITY)}') b
      ANTI JOIN read_parquet('{_sql(ACTIVITY_WALLETS)}') n USING (wallet)
      WHERE {predicate.replace('event_type', 'b.event_type').replace('launchpad', 'b.launchpad')}
    """


def build_quality_lifecycle(force: bool = False) -> Path:
    """Build prior-launch/creator-fee facts without assigning future facts early.

    A creator-fee claim is treated as a defensible success proxy, not token ROI. The
    claim contributes to a candidate only from its own observed timestamp onward.
    """
    if QUALITY_LIFECYCLE.exists() and not force:
        return QUALITY_LIFECYCLE
    if not ACTIVITY_WALLETS.exists():
        raise FileNotFoundError(f"Missing {ACTIVITY_WALLETS}; build the feature store first")
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH raw AS (
            SELECT wallet, token_address, timestamp, event_type,
                   try_cast(cost_usd AS DOUBLE) AS cost_usd,
                   try_cast(quote_amount AS DOUBLE) AS quote_sol
            FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')
            WHERE launchpad='pump' AND event_type IN ('launch','claim_fee')
            UNION ALL
            SELECT b.wallet, b.token_address, b.timestamp, b.event_type,
                   try_cast(b.cost_usd AS DOUBLE), try_cast(b.quote_amount AS DOUBLE)
            FROM read_parquet('{_sql(BOUGHT_ACTIVITY)}') b
            ANTI JOIN read_parquet('{_sql(ACTIVITY_WALLETS)}') n USING (wallet)
            WHERE b.launchpad='pump' AND b.event_type IN ('launch','claim_fee')
          ), launches AS (
            SELECT wallet, token_address, min(timestamp) AS launch_time
            FROM raw WHERE event_type='launch' AND token_address IS NOT NULL
            GROUP BY 1,2
          ), claims AS (
            SELECT wallet, token_address, min(timestamp) AS first_claim_time,
                   count(*) AS claim_event_count,
                   sum(coalesce(cost_usd,0)) AS claim_fee_usd_sum,
                   sum(coalesce(quote_sol,0)) AS claim_fee_sol_sum,
                   max(coalesce(cost_usd,0)) AS claim_fee_usd_max
            FROM raw WHERE event_type='claim_fee' AND token_address IS NOT NULL
            GROUP BY 1,2
          )
          SELECT l.wallet, l.token_address, l.launch_time,
                 CASE WHEN c.first_claim_time>l.launch_time THEN c.first_claim_time END AS first_claim_time,
                 CASE WHEN c.first_claim_time>l.launch_time THEN c.claim_event_count ELSE 0 END AS claim_event_count,
                 CASE WHEN c.first_claim_time>l.launch_time THEN c.claim_fee_usd_sum ELSE 0 END AS claim_fee_usd_sum,
                 CASE WHEN c.first_claim_time>l.launch_time THEN c.claim_fee_sol_sum ELSE 0 END AS claim_fee_sol_sum,
                 CASE WHEN c.first_claim_time>l.launch_time THEN c.claim_fee_usd_max ELSE 0 END AS claim_fee_usd_max
          FROM launches l LEFT JOIN claims c USING (wallet, token_address)
          ORDER BY wallet, launch_time
        ) TO '{_sql(QUALITY_LIFECYCLE)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {QUALITY_LIFECYCLE} in {time.monotonic() - started:.1f}s")
    return QUALITY_LIFECYCLE


def build_quality_state(force: bool = False) -> Path:
    """Create a wallet-time cumulative state whose updates occur when observable."""
    build_quality_lifecycle(force=force)
    if QUALITY_STATE.exists() and not force:
        return QUALITY_STATE
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH raw_claims AS (
            SELECT wallet, token_address, timestamp,
                   coalesce(try_cast(cost_usd AS DOUBLE),0) AS claim_usd,
                   coalesce(try_cast(quote_amount AS DOUBLE),0) AS claim_sol
            FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')
            WHERE launchpad='pump' AND event_type='claim_fee'
            UNION ALL
            SELECT b.wallet, b.token_address, b.timestamp,
                   coalesce(try_cast(b.cost_usd AS DOUBLE),0),
                   coalesce(try_cast(b.quote_amount AS DOUBLE),0)
            FROM read_parquet('{_sql(BOUGHT_ACTIVITY)}') b
            ANTI JOIN read_parquet('{_sql(ACTIVITY_WALLETS)}') n USING (wallet)
            WHERE b.launchpad='pump' AND b.event_type='claim_fee'
          ), first_claims AS (
            SELECT wallet, token_address, min(timestamp) AS timestamp
            FROM raw_claims GROUP BY 1,2
          ), updates AS (
            SELECT wallet, launch_time AS timestamp,
                   1::BIGINT AS pump_launches, 0::BIGINT AS claimed_launches,
                   0::BIGINT AS claim_events, 0.0::DOUBLE AS claim_usd,
                   0.0::DOUBLE AS claim_sol, 0.0::DOUBLE AS claim_max_usd,
                   0::BIGINT AS mature_1d_launches, 0::BIGINT AS mature_1d_successes,
                   0::BIGINT AS mature_7d_launches, 0::BIGINT AS mature_7d_successes,
                   0.0::DOUBLE AS decay30_mature_weight,
                   0.0::DOUBLE AS decay30_success_weight,
                   0::UTINYINT AS has_claim_update
            FROM read_parquet('{_sql(QUALITY_LIFECYCLE)}')
            UNION ALL
            SELECT wallet, timestamp, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 1 FROM first_claims
            UNION ALL
            SELECT wallet, timestamp, 0, 0, count(*), sum(claim_usd), sum(claim_sol),
                   max(claim_usd), 0, 0, 0, 0, 0, 0, 1
            FROM raw_claims GROUP BY wallet,timestamp
            UNION ALL
            SELECT wallet, launch_time+{DAY}, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                   exp((launch_time+{DAY}-{EPOCH_2026})/(30.0*{DAY})), 0, 0
            FROM read_parquet('{_sql(QUALITY_LIFECYCLE)}')
            UNION ALL
            SELECT wallet, launch_time+{DAY},
                   0,0,0,0,0,0,0,1,0,0,0,
                   exp((launch_time+{DAY}-{EPOCH_2026})/(30.0*{DAY})),0
            FROM read_parquet('{_sql(QUALITY_LIFECYCLE)}')
            WHERE first_claim_time<=launch_time+{DAY}
            UNION ALL
            SELECT wallet, launch_time+7*{DAY}, 0,0,0,0,0,0,0,0,1,0,0,0,0
            FROM read_parquet('{_sql(QUALITY_LIFECYCLE)}')
            UNION ALL
            SELECT wallet, launch_time+7*{DAY},
                   0,0,0,0,0,0,0,0,0,1,0,0,0
            FROM read_parquet('{_sql(QUALITY_LIFECYCLE)}')
            WHERE first_claim_time<=launch_time+7*{DAY}
          ), per_second AS (
            SELECT wallet,timestamp,
                   sum(pump_launches) pump_launches,
                   sum(claimed_launches) claimed_launches,
                   sum(claim_events) claim_events,
                   sum(claim_usd) claim_usd,
                   sum(claim_sol) claim_sol,
                   max(claim_max_usd) claim_max_usd,
                   sum(mature_1d_launches) mature_1d_launches,
                   sum(mature_1d_successes) mature_1d_successes,
                   sum(mature_7d_launches) mature_7d_launches,
                   sum(mature_7d_successes) mature_7d_successes,
                   sum(decay30_mature_weight) decay30_mature_weight,
                   sum(decay30_success_weight) decay30_success_weight,
                   max(has_claim_update) has_claim_update
            FROM updates GROUP BY 1,2
          )
          SELECT wallet,timestamp,
                 sum(pump_launches) OVER w AS hist_pump_launch_count,
                 sum(claimed_launches) OVER w AS hist_claimed_launch_count,
                 sum(claim_events) OVER w AS hist_claim_event_count,
                 sum(claim_usd) OVER w AS hist_claim_fee_usd_sum,
                 sum(claim_sol) OVER w AS hist_claim_fee_sol_sum,
                 max(claim_max_usd) OVER w AS hist_claim_fee_usd_max,
                 sum(mature_1d_launches) OVER w AS hist_mature_1d_launch_count,
                 sum(mature_1d_successes) OVER w AS hist_mature_1d_success_count,
                 sum(mature_7d_launches) OVER w AS hist_mature_7d_launch_count,
                 sum(mature_7d_successes) OVER w AS hist_mature_7d_success_count,
                 sum(decay30_mature_weight) OVER w AS hist_decay30_mature_weight,
                 sum(decay30_success_weight) OVER w AS hist_decay30_success_weight,
                 max(timestamp) FILTER (WHERE has_claim_update=1) OVER w AS last_claim_time
          FROM per_second
          WINDOW w AS (PARTITION BY wallet ORDER BY timestamp
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          ORDER BY wallet,timestamp
        ) TO '{_sql(QUALITY_STATE)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {QUALITY_STATE} in {time.monotonic() - started:.1f}s")
    return QUALITY_STATE


def build_quality_features(force: bool = False) -> Path:
    build_quality_state(force=force)
    if QUALITY_FEATURES.exists() and not force:
        return QUALITY_FEATURES
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          SELECT f.token_address,
                 (q.wallet IS NULL)::UTINYINT AS quality_history_missing,
                 (coalesce(q.hist_claimed_launch_count,0)>
                    coalesce(q.hist_pump_launch_count,0))::UTINYINT
                   AS quality_launch_history_incomplete,
                 q.timestamp AS quality_state_time,
                 coalesce(q.hist_pump_launch_count,0) AS hist_pump_launch_count,
                 coalesce(q.hist_claimed_launch_count,0) AS hist_claimed_launch_count,
                 coalesce(q.hist_claim_event_count,0) AS hist_claim_event_count,
                 coalesce(q.hist_claim_fee_sol_sum,0) AS hist_claim_fee_sol_sum,
                 coalesce(q.hist_claim_fee_usd_sum,0) AS hist_claim_fee_usd_sum,
                 coalesce(q.hist_claim_fee_usd_max,0) AS hist_claim_fee_usd_max,
                 least(coalesce(q.hist_claimed_launch_count,0) /
                   greatest(coalesce(q.hist_pump_launch_count,0),1),1)
                   AS hist_claimed_launch_fraction,
                 least(coalesce(q.hist_claimed_launch_count,0) /
                   greatest(coalesce(f.prior_deploy_count,0),1),1)
                   AS hist_claimed_launch_per_core_deploy,
                 coalesce(q.hist_claim_fee_usd_sum,0) /
                   greatest(coalesce(q.hist_claimed_launch_count,0),1) AS hist_claim_fee_usd_per_claimed_launch,
                 f.block_time-q.last_claim_time AS seconds_since_claim_fee,
                 coalesce(q.hist_mature_1d_launch_count,0) AS hist_mature_1d_launch_count,
                 coalesce(q.hist_mature_1d_success_count,0) AS hist_mature_1d_success_count,
                 coalesce(q.hist_mature_1d_success_count,0) /
                   greatest(coalesce(q.hist_mature_1d_launch_count,0),1) AS hist_mature_1d_success_fraction,
                 coalesce(q.hist_mature_7d_launch_count,0) AS hist_mature_7d_launch_count,
                 coalesce(q.hist_mature_7d_success_count,0) AS hist_mature_7d_success_count,
                 coalesce(q.hist_mature_7d_success_count,0) /
                   greatest(coalesce(q.hist_mature_7d_launch_count,0),1) AS hist_mature_7d_success_fraction,
                 coalesce(q.hist_claimed_launch_count,0)-coalesce(q7.hist_claimed_launch_count,0)
                   AS hist_claimed_launch_count_7d,
                 coalesce(q.hist_claimed_launch_count,0)-coalesce(q30.hist_claimed_launch_count,0)
                   AS hist_claimed_launch_count_30d,
                 coalesce(q.hist_claim_fee_usd_sum,0)-coalesce(q7.hist_claim_fee_usd_sum,0)
                   AS hist_claim_fee_usd_7d,
                 coalesce(q.hist_claim_fee_usd_sum,0)-coalesce(q30.hist_claim_fee_usd_sum,0)
                   AS hist_claim_fee_usd_30d,
                 coalesce(q.hist_decay30_success_weight,0) /
                   greatest(coalesce(q.hist_decay30_mature_weight,0),1e-12)
                   AS hist_decayed_30d_success_fraction,
                 l.launch_time AS latest_prior_launch_time,
                 CASE WHEN l.first_claim_time<f.block_time THEN 1 ELSE 0 END::UTINYINT
                   AS latest_prior_launch_claimed,
                 CASE WHEN f.block_time>l.launch_time+{DAY}
                            AND (l.first_claim_time IS NULL
                                 OR l.first_claim_time>l.launch_time+{DAY})
                      THEN 1 ELSE 0 END::UTINYINT AS latest_prior_launch_mature_1d_failure,
                 CASE WHEN f.block_time>l.launch_time+7*{DAY}
                            AND (l.first_claim_time IS NULL
                                 OR l.first_claim_time>l.launch_time+7*{DAY})
                      THEN 1 ELSE 0 END::UTINYINT AS latest_prior_launch_mature_7d_failure,
                 CASE WHEN l.first_claim_time<f.block_time
                      THEN l.first_claim_time-l.launch_time END
                   AS latest_prior_launch_claim_latency_seconds
          FROM read_parquet('{_sql(FEATURE_STORE)}') f
          ASOF LEFT JOIN read_parquet('{_sql(QUALITY_STATE)}') q
            ON f.tx_signer=q.wallet AND f.block_time>q.timestamp
          ASOF LEFT JOIN read_parquet('{_sql(QUALITY_STATE)}') q7
            ON f.tx_signer=q7.wallet AND f.block_time-7*{DAY}>q7.timestamp
          ASOF LEFT JOIN read_parquet('{_sql(QUALITY_STATE)}') q30
            ON f.tx_signer=q30.wallet AND f.block_time-30*{DAY}>q30.timestamp
          ASOF LEFT JOIN read_parquet('{_sql(QUALITY_LIFECYCLE)}') l
            ON f.tx_signer=l.wallet AND f.block_time>l.launch_time
        ) TO '{_sql(QUALITY_FEATURES)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {QUALITY_FEATURES} in {time.monotonic() - started:.1f}s")
    return QUALITY_FEATURES


def audit_quality_features() -> dict[str, object]:
    con = _connection("24GB")
    row = con.execute(
        f"""
        SELECT count(*) AS rows, count(DISTINCT token_address) AS tokens,
               count(*)-count(DISTINCT token_address) AS duplicate_keys,
               count(*) FILTER (WHERE quality_state_time IS NOT NULL
                                      AND quality_state_time>=f.block_time) AS future_states,
               count(*) FILTER (WHERE latest_prior_launch_time IS NOT NULL
                                      AND latest_prior_launch_time>=f.block_time) AS future_launches,
               count(*) FILTER (WHERE seconds_since_claim_fee<1) AS invalid_claim_recencies,
               count(*) FILTER (WHERE hist_mature_1d_success_count>hist_mature_1d_launch_count)
                   AS invalid_1d_fractions,
               count(*) FILTER (WHERE hist_mature_7d_success_count>hist_mature_7d_launch_count)
                   AS invalid_7d_fractions,
               count(*) FILTER (WHERE hist_claimed_launch_fraction<0
                                      OR hist_claimed_launch_fraction>1
                                      OR hist_claimed_launch_per_core_deploy<0
                                      OR hist_claimed_launch_per_core_deploy>1
                                      OR hist_decayed_30d_success_fraction<0
                                      OR hist_decayed_30d_success_fraction>1)
                   AS invalid_bounded_fractions,
               sum(quality_launch_history_incomplete)::BIGINT
                   AS incomplete_launch_histories
        FROM read_parquet('{_sql(QUALITY_FEATURES)}') q
        JOIN read_parquet('{_sql(FEATURE_STORE)}') f USING(token_address)
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    result = dict(zip(names, row, strict=True))
    if result["rows"] != 5_076_421 or any(
        result[name]
        for name in (
            "duplicate_keys",
            "future_states",
            "future_launches",
            "invalid_claim_recencies",
            "invalid_1d_fractions",
            "invalid_7d_fractions",
            "invalid_bounded_fractions",
        )
    ):
        raise RuntimeError(f"historical-quality temporal audit failed: {result}")
    return result


def build_dev_sell_lifecycle(force: bool = False) -> Path:
    """Attach the deployer's first observed sell to each of its prior Pump launches."""
    build_quality_lifecycle(force=force)
    if DEV_SELL_LIFECYCLE.exists() and not force:
        return DEV_SELL_LIFECYCLE
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH first_sells AS (
            SELECT wallet,token_address,min(timestamp) AS first_dev_sell_time
            FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')
            WHERE launchpad='pump' AND event_type='sell'
            GROUP BY 1,2
            UNION ALL
            SELECT b.wallet,b.token_address,min(b.timestamp)
            FROM read_parquet('{_sql(BOUGHT_ACTIVITY)}') b
            ANTI JOIN read_parquet('{_sql(ACTIVITY_WALLETS)}') n USING(wallet)
            WHERE b.launchpad='pump' AND b.event_type='sell'
            GROUP BY 1,2
          )
          SELECT l.wallet,l.token_address,l.launch_time,
                 CASE WHEN s.first_dev_sell_time>l.launch_time
                      THEN s.first_dev_sell_time END AS first_dev_sell_time
          FROM read_parquet('{_sql(QUALITY_LIFECYCLE)}') l
          LEFT JOIN first_sells s USING(wallet,token_address)
          ORDER BY l.wallet,l.launch_time
        ) TO '{_sql(DEV_SELL_LIFECYCLE)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {DEV_SELL_LIFECYCLE} in {time.monotonic() - started:.1f}s")
    return DEV_SELL_LIFECYCLE


def build_dev_sell_state(force: bool = False) -> Path:
    build_dev_sell_lifecycle(force=force)
    if DEV_SELL_STATE.exists() and not force:
        return DEV_SELL_STATE
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH updates AS (
            SELECT wallet,first_dev_sell_time AS timestamp,
                   1::BIGINT AS sold_launches,
                   0::BIGINT AS mature_1d_sold,
                   0::BIGINT AS mature_7d_sold,
                   1::UTINYINT AS has_sell_update
            FROM read_parquet('{_sql(DEV_SELL_LIFECYCLE)}')
            WHERE first_dev_sell_time IS NOT NULL
            UNION ALL
            SELECT wallet,launch_time+{DAY},0,1,0,0
            FROM read_parquet('{_sql(DEV_SELL_LIFECYCLE)}')
            WHERE first_dev_sell_time<=launch_time+{DAY}
            UNION ALL
            SELECT wallet,launch_time+7*{DAY},0,0,1,0
            FROM read_parquet('{_sql(DEV_SELL_LIFECYCLE)}')
            WHERE first_dev_sell_time<=launch_time+7*{DAY}
          ), per_second AS (
            SELECT wallet,timestamp,sum(sold_launches) sold_launches,
                   sum(mature_1d_sold) mature_1d_sold,
                   sum(mature_7d_sold) mature_7d_sold,
                   max(has_sell_update) has_sell_update
            FROM updates GROUP BY 1,2
          )
          SELECT wallet,timestamp,
                 sum(sold_launches) OVER w AS hist_dev_sold_launch_count,
                 sum(mature_1d_sold) OVER w AS hist_mature_1d_dev_sold_count,
                 sum(mature_7d_sold) OVER w AS hist_mature_7d_dev_sold_count,
                 max(timestamp) FILTER (WHERE has_sell_update=1) OVER w
                   AS last_dev_sell_time
          FROM per_second
          WINDOW w AS (PARTITION BY wallet ORDER BY timestamp
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          ORDER BY wallet,timestamp
        ) TO '{_sql(DEV_SELL_STATE)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {DEV_SELL_STATE} in {time.monotonic() - started:.1f}s")
    return DEV_SELL_STATE


def _latest_dev_sell_group_sql(lifecycle_relation: str) -> str:
    """Collapse same-wallet/same-second launches without inventing an order.

    The activity source resolves time only to seconds. Transaction hashes identify
    events but do not order them, and the core index does not provide a transaction
    position. One row per wallet-second makes the later ASOF key unique and stores
    the complete symmetric outcome multiset for candidate-time summaries.
    """
    return f"""
      SELECT wallet,launch_time,
             count(*)::BIGINT AS latest_launch_group_size,
             list(first_dev_sell_time) AS first_dev_sell_times
      FROM {lifecycle_relation}
      GROUP BY wallet,launch_time
    """


def _latest_dev_sell_group_feature_expressions(
    candidate_alias: str = "f", group_alias: str = "l"
) -> dict[str, str]:
    """Return the order-invariant latest-launch-group feature expressions."""
    candidate_time = f"{candidate_alias}.block_time"
    launch_time = f"{group_alias}.launch_time"
    sell_times = f"{group_alias}.first_dev_sell_times"
    group_size = f"{group_alias}.latest_launch_group_size"
    observed = (
        f"list_filter({sell_times}, first_sell -> "
        f"first_sell IS NOT NULL AND first_sell<{candidate_time})"
    )
    return {
        "latest_prior_launch_group_dev_sold_fraction": (
            f"coalesce(list_count({observed})::DOUBLE/{group_size},0.0)"
        ),
        "latest_prior_launch_group_mature_1d_no_dev_sell_fraction": (
            f"CASE WHEN {candidate_time}>{launch_time}+{DAY} THEN "
            f"({group_size}-list_count(list_filter({sell_times}, first_sell -> "
            f"first_sell IS NOT NULL AND first_sell<={launch_time}+{DAY})))::DOUBLE/"
            f"{group_size} ELSE 0.0 END"
        ),
        "latest_prior_launch_group_mature_7d_no_dev_sell_fraction": (
            f"CASE WHEN {candidate_time}>{launch_time}+7*{DAY} THEN "
            f"({group_size}-list_count(list_filter({sell_times}, first_sell -> "
            f"first_sell IS NOT NULL AND first_sell<={launch_time}+7*{DAY})))::DOUBLE/"
            f"{group_size} ELSE 0.0 END"
        ),
        "latest_prior_launch_group_dev_sell_latency_median_seconds": (
            f"list_median(list_transform({observed}, first_sell -> "
            f"first_sell-{launch_time}))"
        ),
    }


def build_dev_sell_features(force: bool = False) -> Path:
    build_quality_features(force=force)
    build_dev_sell_state(force=force)
    if DEV_SELL_FEATURES.exists() and not force:
        return DEV_SELL_FEATURES
    con = _connection()
    started = time.monotonic()
    latest_group_sql = _latest_dev_sell_group_sql(
        f"read_parquet('{_sql(DEV_SELL_LIFECYCLE)}')"
    )
    latest_expressions = _latest_dev_sell_group_feature_expressions()
    con.execute(
        f"""
        COPY (
          WITH latest_groups AS ({latest_group_sql})
          SELECT f.token_address,d.timestamp AS dev_sell_state_time,
                 coalesce(d.hist_dev_sold_launch_count,0)
                   AS hist_dev_sold_launch_count,
                 least(coalesce(d.hist_dev_sold_launch_count,0) /
                   greatest(q.hist_pump_launch_count,1),1)
                   AS hist_dev_sold_launch_fraction,
                 f.block_time-d.last_dev_sell_time AS seconds_since_prior_dev_sell,
                 coalesce(d.hist_mature_1d_dev_sold_count,0)
                   AS hist_mature_1d_dev_sold_count,
                 coalesce(d.hist_mature_1d_dev_sold_count,0) /
                   greatest(q.hist_mature_1d_launch_count,1)
                   AS hist_mature_1d_dev_sold_fraction,
                 coalesce(d.hist_mature_7d_dev_sold_count,0)
                   AS hist_mature_7d_dev_sold_count,
                 coalesce(d.hist_mature_7d_dev_sold_count,0) /
                   greatest(q.hist_mature_7d_launch_count,1)
                   AS hist_mature_7d_dev_sold_fraction,
                 coalesce(d.hist_dev_sold_launch_count,0)-
                   coalesce(d7.hist_dev_sold_launch_count,0)
                   AS hist_dev_sold_launch_count_7d,
                 coalesce(d.hist_dev_sold_launch_count,0)-
                   coalesce(d30.hist_dev_sold_launch_count,0)
                   AS hist_dev_sold_launch_count_30d,
                 l.launch_time AS latest_dev_sell_launch_time,
                 l.latest_launch_group_size AS latest_dev_sell_group_size,
                 {latest_expressions['latest_prior_launch_group_dev_sold_fraction']}
                   AS latest_prior_launch_group_dev_sold_fraction,
                 {latest_expressions['latest_prior_launch_group_mature_1d_no_dev_sell_fraction']}
                   AS latest_prior_launch_group_mature_1d_no_dev_sell_fraction,
                 {latest_expressions['latest_prior_launch_group_mature_7d_no_dev_sell_fraction']}
                   AS latest_prior_launch_group_mature_7d_no_dev_sell_fraction,
                 {latest_expressions['latest_prior_launch_group_dev_sell_latency_median_seconds']}
                   AS latest_prior_launch_group_dev_sell_latency_median_seconds
          FROM read_parquet('{_sql(FEATURE_STORE)}') f
          JOIN read_parquet('{_sql(QUALITY_FEATURES)}') q USING(token_address)
          ASOF LEFT JOIN read_parquet('{_sql(DEV_SELL_STATE)}') d
            ON f.tx_signer=d.wallet AND f.block_time>d.timestamp
          ASOF LEFT JOIN read_parquet('{_sql(DEV_SELL_STATE)}') d7
            ON f.tx_signer=d7.wallet AND f.block_time-7*{DAY}>d7.timestamp
          ASOF LEFT JOIN read_parquet('{_sql(DEV_SELL_STATE)}') d30
            ON f.tx_signer=d30.wallet AND f.block_time-30*{DAY}>d30.timestamp
          ASOF LEFT JOIN latest_groups l
            ON f.tx_signer=l.wallet AND f.block_time>l.launch_time
          ORDER BY f.block_time,f.token_address
        ) TO '{_sql(DEV_SELL_FEATURES)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {DEV_SELL_FEATURES} in {time.monotonic() - started:.1f}s")
    return DEV_SELL_FEATURES


def audit_dev_sell_features() -> dict[str, int]:
    con = _connection("24GB")
    row = con.execute(
        f"""
        SELECT count(*) AS rows,count(DISTINCT d.token_address) AS tokens,
               count(*)-count(DISTINCT d.token_address) AS duplicate_keys,
               count(*) FILTER (WHERE dev_sell_state_time>=f.block_time) AS future_states,
               count(*) FILTER (WHERE latest_dev_sell_launch_time>=f.block_time) AS future_launches,
               count(*) FILTER (WHERE seconds_since_prior_dev_sell<1) AS invalid_recencies,
               count(*) FILTER (WHERE latest_dev_sell_group_size>1) AS tied_latest_group_rows,
               count(*) FILTER (
                 WHERE latest_prior_launch_group_dev_sold_fraction NOT BETWEEN 0 AND 1
                    OR latest_prior_launch_group_mature_1d_no_dev_sell_fraction NOT BETWEEN 0 AND 1
                    OR latest_prior_launch_group_mature_7d_no_dev_sell_fraction NOT BETWEEN 0 AND 1
                    OR latest_prior_launch_group_dev_sell_latency_median_seconds<1
               ) AS invalid_latest_group_summaries,
               count(*) FILTER (
                 WHERE hist_mature_1d_dev_sold_count>q.hist_mature_1d_launch_count
                    OR hist_mature_7d_dev_sold_count>q.hist_mature_7d_launch_count
                    OR hist_dev_sold_launch_fraction NOT BETWEEN 0 AND 1
                    OR hist_mature_1d_dev_sold_fraction NOT BETWEEN 0 AND 1
                    OR hist_mature_7d_dev_sold_fraction NOT BETWEEN 0 AND 1
               ) AS invalid_fractions
        FROM read_parquet('{_sql(DEV_SELL_FEATURES)}') d
        JOIN read_parquet('{_sql(FEATURE_STORE)}') f USING(token_address)
        JOIN read_parquet('{_sql(QUALITY_FEATURES)}') q USING(token_address)
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    result = {name: int(value) for name, value in zip(names, row, strict=True)}
    if result["rows"] != 5_076_421 or any(
        result[name]
        for name in (
            "duplicate_keys",
            "future_states",
            "future_launches",
            "invalid_recencies",
            "invalid_fractions",
            "invalid_latest_group_summaries",
        )
    ):
        raise RuntimeError(f"developer-sell temporal audit failed: {result}")
    return result


def audit_dev_sell_tie_contract() -> dict[str, object]:
    """Prove why same-second launch groups cannot be temporally ordered."""
    con = _connection("30GB")
    activity_columns = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')"
    ).fetch_df().column_name.tolist()
    index_columns = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_sql(NOT_BOUGHT_INDEX)}')"
    ).fetch_df().column_name.tolist()
    row = con.execute(
        f"""
        WITH raw_launch AS (
          SELECT wallet,token_address,timestamp,tx_hash
          FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')
          WHERE launchpad='pump' AND event_type='launch' AND token_address IS NOT NULL
          UNION ALL
          SELECT b.wallet,b.token_address,b.timestamp,b.tx_hash
          FROM read_parquet('{_sql(BOUGHT_ACTIVITY)}') b
          ANTI JOIN read_parquet('{_sql(ACTIVITY_WALLETS)}') n USING(wallet)
          WHERE b.launchpad='pump' AND b.event_type='launch'
            AND b.token_address IS NOT NULL
        ), first_time AS (
          SELECT wallet,token_address,min(timestamp) AS launch_time
          FROM raw_launch GROUP BY 1,2
        ), launches AS (
          SELECT f.wallet,f.token_address,f.launch_time,min(r.tx_hash) AS launch_tx_hash
          FROM first_time f
          JOIN raw_launch r ON f.wallet=r.wallet AND f.token_address=r.token_address
                           AND f.launch_time=r.timestamp
          GROUP BY 1,2,3
        ), ties AS (
          SELECT wallet,launch_time,count(*) AS tie_size
          FROM launches GROUP BY 1,2 HAVING count(*)>1
        ), deployment_index AS (
          SELECT tx_hash,blockTime,blockSlot,token_address,tx_signer
          FROM read_parquet('{_sql(NOT_BOUGHT_INDEX)}')
          UNION ALL
          SELECT tx_hash,blockTime,blockSlot,token_address,tx_signer
          FROM read_parquet('{_sql(BOUGHT_INDEX)}')
        ), joined AS (
          SELECT l.*,t.tie_size,i.tx_hash AS index_tx_hash,i.blockTime,i.blockSlot,
                 i.tx_signer,(i.token_address IS NOT NULL) AS mapped
          FROM launches l JOIN ties t USING(wallet,launch_time)
          LEFT JOIN deployment_index i USING(token_address)
        ), group_audit AS (
          SELECT wallet,launch_time,max(tie_size) AS tie_size,
                 count(*) FILTER (WHERE mapped) AS mapped,
                 count(DISTINCT blockSlot) AS distinct_slots
          FROM joined GROUP BY 1,2
        )
        SELECT
          (SELECT count(*) FROM group_audit) AS tie_groups,
          (SELECT sum(tie_size) FROM group_audit) AS tied_launches,
          (SELECT max(tie_size) FROM group_audit) AS max_tie_size,
          (SELECT count(*) FROM joined WHERE mapped) AS mapped_launches,
          (SELECT count(*) FROM joined WHERE mapped AND launch_tx_hash=index_tx_hash)
            AS mapped_tx_hash_matches,
          (SELECT count(*) FROM joined WHERE mapped AND launch_time=blockTime)
            AS mapped_time_matches,
          (SELECT count(*) FROM joined WHERE mapped AND wallet=tx_signer)
            AS mapped_signer_matches,
          (SELECT count(*) FROM group_audit WHERE mapped=tie_size)
            AS fully_mapped_groups,
          (SELECT count(*) FROM group_audit
             WHERE mapped=tie_size AND distinct_slots=tie_size)
            AS fully_resolved_unique_slot_groups,
          (SELECT count(*) FROM group_audit
             WHERE mapped=tie_size AND distinct_slots<tie_size)
            AS fully_mapped_groups_with_same_slot_ties,
          (SELECT count(*) FROM group_audit WHERE mapped<tie_size)
            AS partially_mapped_groups
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    counts = {name: int(value) for name, value in zip(names, row, strict=True)}
    if counts != {
        "tie_groups": 157_071,
        "tied_launches": 518_390,
        "max_tie_size": 84,
        "mapped_launches": 516_222,
        "mapped_tx_hash_matches": 516_222,
        "mapped_time_matches": 516_222,
        "mapped_signer_matches": 516_222,
        "fully_mapped_groups": 156_271,
        "fully_resolved_unique_slot_groups": 18_391,
        "fully_mapped_groups_with_same_slot_ties": 137_880,
        "partially_mapped_groups": 800,
    }:
        raise RuntimeError(f"equal-timestamp source audit changed: {counts}")
    return {
        **counts,
        "activity_temporal_fields": [
            name for name in ("timestamp", "blockSlot", "transactionIndex")
            if name in activity_columns
        ],
        "activity_identifier_fields": [name for name in ("tx_hash",) if name in activity_columns],
        "deployment_index_temporal_fields": [
            name for name in ("blockTime", "blockSlot", "transactionIndex")
            if name in index_columns
        ],
        "contract": "Group all launches sharing (wallet, launch_time); use symmetric fractions and the observed-latency median. Transaction hash and file order are identifiers/storage order, not temporal order.",
    }


def build_metadata_features(force: bool = False) -> Path:
    """Add strict-prior metadata reuse and signed-message construction features."""
    if METADATA_FEATURES.exists() and not force:
        return METADATA_FEATURES
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH b AS (
            SELECT * FROM read_parquet('{_sql(BASE_FEATURES)}')
          ), uri_groups AS (
            SELECT uri,block_time,count(*) AS n
            FROM b WHERE uri IS NOT NULL AND uri<>'' GROUP BY 1,2
          ), uri_history AS (
            SELECT *,coalesce(sum(n) OVER (PARTITION BY uri ORDER BY block_time
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS prior_uri_count,
              lag(block_time) OVER (PARTITION BY uri ORDER BY block_time) AS prior_uri_time
            FROM uri_groups
          ), signer_uri_groups AS (
            SELECT tx_signer,uri,block_time,count(*) AS n
            FROM b WHERE uri IS NOT NULL AND uri<>'' GROUP BY 1,2,3
          ), signer_uri_history AS (
            SELECT *,coalesce(sum(n) OVER (PARTITION BY tx_signer,uri ORDER BY block_time
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS signer_prior_uri_count
            FROM signer_uri_groups
          ), host_groups AS (
            SELECT uri_host,block_time,count(*) AS n
            FROM b WHERE uri_host IS NOT NULL AND uri_host<>'' GROUP BY 1,2
          ), host_history AS (
            SELECT *,coalesce(sum(n) OVER (PARTITION BY uri_host ORDER BY block_time
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS prior_uri_host_count
            FROM host_groups
          ), signer_host_groups AS (
            SELECT tx_signer,uri_host,block_time,count(*) AS n
            FROM b WHERE uri_host IS NOT NULL AND uri_host<>'' GROUP BY 1,2,3
          ), signer_host_history AS (
            SELECT *,coalesce(sum(n) OVER (PARTITION BY tx_signer,uri_host ORDER BY block_time
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS signer_prior_uri_host_count
            FROM signer_host_groups
          ), pair_groups AS (
            SELECT name_normalized,symbol_normalized,block_time,count(*) AS n
            FROM b WHERE name_normalized IS NOT NULL AND symbol_normalized IS NOT NULL
            GROUP BY 1,2,3
          ), pair_history AS (
            SELECT *,coalesce(sum(n) OVER (
              PARTITION BY name_normalized,symbol_normalized ORDER BY block_time
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS prior_name_symbol_count
            FROM pair_groups
          ), signer_pair_groups AS (
            SELECT tx_signer,name_normalized,symbol_normalized,block_time,count(*) AS n
            FROM b WHERE name_normalized IS NOT NULL AND symbol_normalized IS NOT NULL
            GROUP BY 1,2,3,4
          ), signer_pair_history AS (
            SELECT *,coalesce(sum(n) OVER (
              PARTITION BY tx_signer,name_normalized,symbol_normalized ORDER BY block_time
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0)
              AS signer_prior_name_symbol_count
            FROM signer_pair_groups
          )
          SELECT b.token_address,
                 coalesce(u.prior_uri_count,0) AS prior_uri_count,
                 b.block_time-u.prior_uri_time AS seconds_since_prior_uri,
                 coalesce(su.signer_prior_uri_count,0) AS signer_prior_uri_count,
                 coalesce(h.prior_uri_host_count,0) AS prior_uri_host_count,
                 coalesce(sh.signer_prior_uri_host_count,0) AS signer_prior_uri_host_count,
                 coalesce(p.prior_name_symbol_count,0) AS prior_name_symbol_count,
                 coalesce(sp.signer_prior_name_symbol_count,0) AS signer_prior_name_symbol_count,
                 coalesce(b.compute_unit_limit,0)*
                   coalesce(b.compute_unit_price_micro_lamports,0)/1000000.0
                   AS priority_fee_intent_lamports,
                 CASE WHEN b.has_dev_buy=1
                      THEN b.dev_buy_instruction_index-b.create_instruction_index END
                   AS dev_buy_instruction_gap,
                 (b.has_dev_buy=1 AND b.dev_buy_instruction_index=b.create_instruction_index+1)::UTINYINT
                   AS dev_buy_immediately_after_create,
                 b.create_instruction_index/greatest(b.n_message_instructions-1,1.0)
                   AS create_instruction_fraction,
                 b.system_transfer_sol/greatest(b.n_system_transfers,1)
                   AS system_transfer_mean_sol
          FROM b
          LEFT JOIN uri_history u USING(uri,block_time)
          LEFT JOIN signer_uri_history su USING(tx_signer,uri,block_time)
          LEFT JOIN host_history h USING(uri_host,block_time)
          LEFT JOIN signer_host_history sh USING(tx_signer,uri_host,block_time)
          LEFT JOIN pair_history p USING(name_normalized,symbol_normalized,block_time)
          LEFT JOIN signer_pair_history sp
            USING(tx_signer,name_normalized,symbol_normalized,block_time)
        ) TO '{_sql(METADATA_FEATURES)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {METADATA_FEATURES} in {time.monotonic() - started:.1f}s")
    return METADATA_FEATURES


def audit_metadata_features() -> dict[str, int]:
    con = _connection("24GB")
    row = con.execute(
        f"""
        SELECT count(*) AS rows,count(DISTINCT token_address) AS tokens,
               count(*)-count(DISTINCT token_address) duplicate_keys,
               count(*) FILTER (WHERE seconds_since_prior_uri<=0) invalid_uri_recency,
               count(*) FILTER (WHERE prior_uri_count<0 OR signer_prior_uri_count<0
                 OR prior_uri_host_count<0 OR signer_prior_uri_host_count<0
                 OR prior_name_symbol_count<0 OR signer_prior_name_symbol_count<0)
                 negative_prior_counts
        FROM read_parquet('{_sql(METADATA_FEATURES)}')
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    result = {name: int(value) for name, value in zip(names, row, strict=True)}
    if result["rows"] != 5_076_421 or any(
        result[name]
        for name in ("duplicate_keys", "invalid_uri_recency", "negative_prior_counts")
    ):
        raise RuntimeError(f"metadata temporal audit failed: {result}")
    return result


def run_signed_message_metadata_experiment(force: bool = False) -> dict[str, object]:
    """Test strict-prior metadata reuse and unused signed-message structure."""
    ensure_output_dirs()
    build_metadata_features(force=force)
    audit = audit_metadata_features()
    baseline_numeric, categorical = _available_features(FEATURE_STORE)
    combined_numeric = baseline_numeric + METADATA_REUSE_FEATURES
    stride = 2
    windows = {
        "april": {
            "train": f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{APRIL_START} "
            f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
            "validation": f"f.block_time>={APRIL_START} AND f.block_time<{MAY_START}",
        },
        "may": {
            "train": f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{MAY_START} "
            f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
            "validation": f"f.block_time>={MAY_START} AND f.block_time<{JUNE_START}",
        },
    }
    output: dict[str, object] = {
        "hypothesis": "Strict-prior metadata reuse and signed-message construction add signal beyond the control feature dictionary.",
        "leakage_policy": "Reuse counts include deployments at strictly earlier block_time only; all construction fields come from the signed deployment message.",
        "temporal_audit": audit,
        "windows": {},
    }
    may_models: dict[str, ModelBundle] = {}
    may_thresholds: dict[str, float] = {}
    for period, bounds in windows.items():
        train = _joined_frame(
            bounds["train"], combined_numeric, categorical, metadata=True
        )
        validation = _joined_frame(
            bounds["validation"], combined_numeric, categorical, metadata=True
        )
        baseline = fit_lightgbm(train, baseline_numeric, categorical, stride)
        family = fit_lightgbm(train, METADATA_REUSE_FEATURES, [], stride)
        combined = fit_lightgbm(train, combined_numeric, categorical, stride)
        baseline_result, _, baseline_threshold = _model_summary(baseline, validation)
        family_result, _, _ = _model_summary(family, validation)
        combined_result, _, combined_threshold = _model_summary(combined, validation)
        output["windows"][period] = {  # type: ignore[index]
            "train_rows": int(len(train)),
            "train_positives": int(train.label.sum()),
            "validation_rows": int(len(validation)),
            "validation_positives": int(validation.label.sum()),
            "baseline": baseline_result,
            "metadata_message_only": family_result,
            "baseline_plus_metadata_message": combined_result,
            "combined_minus_baseline": {
                key: float(combined_result[key] - baseline_result[key])
                for key in ("pr_auc", "precision", "recall", "f1")
            },
            "group_permutation_pr_auc_drop": _group_permutation_drop(
                combined, validation, METADATA_REUSE_FEATURES
            ),
            "feature_gain_top15": _gain_importance(
                combined, METADATA_REUSE_FEATURES
            ),
        }
        if period == "may":
            may_models = {"baseline": baseline, "combined": combined}
            may_thresholds = {
                "baseline": baseline_threshold,
                "combined": combined_threshold,
            }
    april_delta = float(
        output["windows"]["april"]["combined_minus_baseline"]["pr_auc"]  # type: ignore[index]
    )
    may_delta = float(
        output["windows"]["may"]["combined_minus_baseline"]["pr_auc"]  # type: ignore[index]
    )
    output["pre_june_decision"] = {
        "status": "KEEP" if april_delta > 0.002 and may_delta > 0.002 else "DROP",
        "rule": "KEEP only for >0.002 PR-AUC lift in both April and May.",
        "april_pr_auc_delta": april_delta,
        "may_pr_auc_delta": may_delta,
    }
    june = _joined_frame(
        f"f.block_time>={JUNE_START}", combined_numeric, categorical, metadata=True
    )
    y = june.label.to_numpy(dtype=np.uint8)
    june_results: dict[str, object] = {}
    for name, model in may_models.items():
        score = predict(model, june)
        june_results[name] = {
            **metrics(y, score, may_thresholds[name]),
            "top_k": _rank_summary(y, score),
        }
    output["june_reporting_only"] = june_results
    METADATA_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def build_claim_outcomes(force: bool = False) -> Path:
    """Build post-deployment creator-fee outcomes for evaluation/Model B only."""
    if CLAIM_OUTCOMES.exists() and not force:
        return CLAIM_OUTCOMES
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH claims AS (
            SELECT wallet,token_address,timestamp,
                   coalesce(try_cast(cost_usd AS DOUBLE),0) AS claim_usd,
                   coalesce(try_cast(quote_amount AS DOUBLE),0) AS claim_sol
            FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')
            WHERE launchpad='pump' AND event_type='claim_fee'
            UNION ALL
            SELECT b.wallet,b.token_address,b.timestamp,
                   coalesce(try_cast(b.cost_usd AS DOUBLE),0),
                   coalesce(try_cast(b.quote_amount AS DOUBLE),0)
            FROM read_parquet('{_sql(BOUGHT_ACTIVITY)}') b
            ANTI JOIN read_parquet('{_sql(ACTIVITY_WALLETS)}') n USING(wallet)
            WHERE b.launchpad='pump' AND b.event_type='claim_fee'
          )
          SELECT f.token_address,
                 min(c.timestamp) AS first_creator_fee_claim_time,
                 count(c.timestamp) FILTER (WHERE c.timestamp<=f.block_time+{DAY}) AS creator_fee_claim_events_1d,
                 count(c.timestamp) AS creator_fee_claim_events_7d,
                 coalesce(sum(c.claim_usd) FILTER (WHERE c.timestamp<=f.block_time+{DAY}),0)
                   AS creator_fee_usd_1d,
                 coalesce(sum(c.claim_usd),0) AS creator_fee_usd_7d,
                 coalesce(sum(c.claim_sol) FILTER (WHERE c.timestamp<=f.block_time+{DAY}),0)
                   AS creator_fee_sol_1d,
                 coalesce(sum(c.claim_sol),0) AS creator_fee_sol_7d
          FROM read_parquet('{_sql(FEATURE_STORE)}') f
          LEFT JOIN claims c ON c.wallet=f.tx_signer AND c.token_address=f.token_address
                            AND c.timestamp>f.block_time
                            AND c.timestamp<=f.block_time+7*{DAY}
          GROUP BY f.token_address
        ) TO '{_sql(CLAIM_OUTCOMES)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {CLAIM_OUTCOMES} in {time.monotonic() - started:.1f}s")
    return CLAIM_OUTCOMES


def _joined_frame(
    predicate: str,
    numeric: list[str],
    categorical: list[str],
    *,
    quality: bool = False,
    dev_sell: bool = False,
    metadata: bool = False,
    outcome: bool = False,
) -> pd.DataFrame:
    feature_store_columns = set(
        duckdb.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql(FEATURE_STORE)}')"
        ).fetchdf().column_name
    )
    extra_sources: list[tuple[str, Path, set[str]]] = []
    if quality:
        build_quality_features()
        names = set(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql(QUALITY_FEATURES)}')"
            ).fetchdf().column_name
        )
        extra_sources.append(("q", QUALITY_FEATURES, names))
    if dev_sell:
        build_dev_sell_features()
        names = set(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql(DEV_SELL_FEATURES)}')"
            ).fetchdf().column_name
        )
        extra_sources.append(("s", DEV_SELL_FEATURES, names))
    if metadata:
        build_metadata_features()
        names = set(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql(METADATA_FEATURES)}')"
            ).fetchdf().column_name
        )
        extra_sources.append(("m", METADATA_FEATURES, names))
    if outcome:
        build_claim_outcomes()
        names = set(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql(CLAIM_OUTCOMES)}')"
            ).fetchdf().column_name
        )
        extra_sources.append(("o", CLAIM_OUTCOMES, names))

    columns = list(
        dict.fromkeys(
            ["token_address", "tx_hash", "tx_signer", "block_time", "label"]
            + numeric
            + categorical
            + (["creator_fee_claim_events_7d", "creator_fee_usd_7d"] if outcome else [])
        )
    )
    base_columns = [column for column in columns if column in feature_store_columns]
    missing_columns = [column for column in columns if column not in feature_store_columns]
    available_extra = set().union(*(names for _, _, names in extra_sources)) if extra_sources else set()
    unavailable = sorted(set(missing_columns) - available_extra)
    if unavailable:
        raise KeyError(f"columns unavailable from requested feature joins: {unavailable}")
    con = _connection("30GB")
    # Load the control rows first. The later keyed augmentation explicitly preserves
    # this order, preventing a join-order change from masquerading as model lift.
    frame = con.execute(
        f"SELECT {', '.join('f.' + chr(34) + column + chr(34) for column in base_columns)} "
        f"FROM read_parquet(?) f WHERE {predicate} "
        "ORDER BY f.block_time,f.token_address",
        [str(FEATURE_STORE)],
    ).fetch_df()
    if not missing_columns:
        return frame
    wanted = frame[["token_address"]].copy()
    wanted["_row_order"] = np.arange(len(wanted), dtype=np.int64)
    con.register("wanted_rows", wanted)
    joins = []
    select_extra = []
    for alias, path, names in extra_sources:
        used = [column for column in missing_columns if column in names]
        if not used:
            continue
        joins.append(f"LEFT JOIN read_parquet('{_sql(path)}') {alias} USING(token_address)")
        select_extra.extend(f'{alias}."{column}"' for column in used)
    augmented = con.execute(
        f"SELECT w.token_address,w._row_order,{','.join(select_extra)} "
        f"FROM wanted_rows w {' '.join(joins)} ORDER BY w._row_order"
    ).fetch_df()
    if len(augmented) != len(frame) or augmented.token_address.duplicated().any():
        raise RuntimeError("extra-feature augmentation changed control row cardinality")
    for column in missing_columns:
        frame[column] = augmented[column].to_numpy()
    return frame


def _rank_summary(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    k = int(y.sum())
    if not k:
        return {"k": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "lift_at_k": 0.0}
    order = np.argpartition(score, -k)[-k:]
    positives = int(y[order].sum())
    prevalence = float(y.mean())
    precision = positives / k
    return {
        "k": k,
        "positives_at_k": positives,
        "precision_at_k": precision,
        "recall_at_k": positives / int(y.sum()),
        "lift_at_k": precision / prevalence,
    }


def _model_summary(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[dict[str, object], np.ndarray, float]:
    y = frame.label.to_numpy(dtype=np.uint8)
    score = predict(bundle, frame)
    threshold = _thresholds(y, score)["max_f1"]
    result: dict[str, object] = metrics(y, score, threshold)
    result["top_k"] = _rank_summary(y, score)
    return result, score, threshold


def _daily_pr_auc_stability(
    period: str,
    validation: pd.DataFrame,
    baseline_score: np.ndarray,
    combined_score: np.ndarray,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    dates = pd.to_datetime(validation.block_time, unit="s", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    rows: list[dict[str, object]] = []
    for date in sorted(dates.unique()):
        mask = dates.eq(date).to_numpy()
        y = validation.loc[mask, "label"].to_numpy(dtype=np.uint8)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        baseline = float(average_precision_score(y, baseline_score[mask]))
        combined = float(average_precision_score(y, combined_score[mask]))
        rows.append(
            {
                "period": period,
                "date_utc": date,
                "rows": int(mask.sum()),
                "positives": int(y.sum()),
                "baseline_pr_auc": baseline,
                "combined_pr_auc": combined,
                "pr_auc_delta": combined - baseline,
            }
        )
    deltas = np.array([row["pr_auc_delta"] for row in rows], dtype=float)
    return {
        "days": int(len(deltas)),
        "positive_delta_days": int((deltas > 0).sum()),
        "positive_delta_share": float((deltas > 0).mean()),
        "mean_daily_delta": float(deltas.mean()),
        "median_daily_delta": float(np.median(deltas)),
        "q25_daily_delta": float(np.quantile(deltas, 0.25)),
        "q75_daily_delta": float(np.quantile(deltas, 0.75)),
        "min_daily_delta": float(deltas.min()),
        "max_daily_delta": float(deltas.max()),
    }, rows


def _group_permutation_drop(
    bundle: ModelBundle,
    validation: pd.DataFrame,
    features: list[str],
) -> float:
    sample = validation.sample(min(200_000, len(validation)), random_state=20260811).copy()
    y = sample.label.to_numpy(dtype=np.uint8)
    baseline = average_precision_score(y, predict(bundle, sample))
    permutation = np.random.default_rng(20260811).permutation(len(sample))
    original = {feature: sample[feature].copy() for feature in features}
    for feature in features:
        sample[feature] = original[feature].to_numpy()[permutation]
    return float(baseline - average_precision_score(y, predict(bundle, sample)))


def _gain_importance(bundle: ModelBundle, prefix_features: list[str]) -> list[dict[str, object]]:
    model = bundle.model.named_steps["model"]
    names = bundle.feature_names_out
    values = model.feature_importances_
    rows = [
        {"feature": str(name), "gain": int(value)}
        for name, value in zip(names, values, strict=True)
        if any(str(name) == feature or str(name).startswith(feature + "_") for feature in prefix_features)
    ]
    return sorted(rows, key=lambda item: item["gain"], reverse=True)[:15]


def run_historical_outcome_experiment(force: bool = False) -> dict[str, object]:
    ensure_output_dirs()
    build_quality_features(force=force)
    temporal_audit = audit_quality_features()
    baseline_numeric, categorical = _available_features(FEATURE_STORE)
    all_numeric = baseline_numeric + HISTORICAL_OUTCOME_FEATURES
    stride = 2
    windows = {
        "april": {
            "train": f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{APRIL_START} "
            f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
            "validation": f"f.block_time>={APRIL_START} AND f.block_time<{MAY_START}",
        },
        "may": {
            "train": f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{MAY_START} "
            f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
            "validation": f"f.block_time>={MAY_START} AND f.block_time<{JUNE_START}",
        },
    }
    output: dict[str, object] = {
        "hypothesis": "Prior creator-fee-earning launches add independent point-in-time quality signal beyond generic wallet activity.",
        "success_proxy": "A Pump creator-fee claim is an observed prior economic outcome, not full token ROI or migration.",
        "temporal_audit": temporal_audit,
        "windows": {},
    }
    may_models: dict[str, object] = {}
    may_thresholds: dict[str, float] = {}
    daily_stability_rows: list[dict[str, object]] = []
    for window, bounds in windows.items():
        train = _joined_frame(
            bounds["train"], all_numeric, categorical, quality=True
        )
        validation = _joined_frame(
            bounds["validation"], all_numeric, categorical, quality=True
        )
        baseline = fit_lightgbm(train, baseline_numeric, categorical, stride)
        quality_only = fit_lightgbm(train, HISTORICAL_OUTCOME_FEATURES, [], stride)
        combined = fit_lightgbm(train, all_numeric, categorical, stride)
        baseline_result, baseline_score, baseline_threshold = _model_summary(baseline, validation)
        quality_result, _, quality_threshold = _model_summary(quality_only, validation)
        combined_result, combined_score, combined_threshold = _model_summary(combined, validation)
        daily_stability, daily_rows = _daily_pr_auc_stability(
            window, validation, baseline_score, combined_score
        )
        daily_stability_rows.extend(daily_rows)
        window_result = {
            "train_rows": int(len(train)),
            "train_positives": int(train.label.sum()),
            "validation_rows": int(len(validation)),
            "validation_positives": int(validation.label.sum()),
            "baseline": baseline_result,
            "historical_outcomes_only": quality_result,
            "baseline_plus_historical_outcomes": combined_result,
            "combined_minus_baseline": {
                "pr_auc": float(combined_result["pr_auc"] - baseline_result["pr_auc"]),
                "precision": float(combined_result["precision"] - baseline_result["precision"]),
                "recall": float(combined_result["recall"] - baseline_result["recall"]),
                "f1": float(combined_result["f1"] - baseline_result["f1"]),
            },
            "quality_group_permutation_pr_auc_drop": _group_permutation_drop(
                combined, validation, HISTORICAL_OUTCOME_FEATURES
            ),
            "quality_feature_gain_top15": _gain_importance(combined, HISTORICAL_OUTCOME_FEATURES),
            "daily_pr_auc_stability": daily_stability,
        }
        output["windows"][window] = window_result  # type: ignore[index]
        if window == "may":
            may_models = {"baseline": baseline, "combined": combined}
            may_thresholds = {"baseline": baseline_threshold, "combined": combined_threshold}
            joblib.dump(combined.model, ARTIFACTS / "models" / "third_pass_quality_model.joblib")
            QUALITY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(combined.model, QUALITY_MODEL_DIR / "final_model.joblib")
            (QUALITY_MODEL_DIR / "model_features.json").write_text(
                json.dumps(
                    {
                        "numeric_features": combined.numeric_features,
                        "categorical_features": combined.categorical_features,
                        "transformed_feature_names": combined.feature_names_out,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            (QUALITY_MODEL_DIR / "operating_point.json").write_text(
                json.dumps(
                    {
                        "model": "quality_augmented_active_era_lightgbm",
                        "selection_policy": "fixed score threshold maximizing F1 on May validation",
                        "threshold": combined_threshold,
                        "selected_pre_june_only": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            pd.DataFrame(
                {
                    "token_address": validation.token_address,
                    "label": validation.label.astype("uint8"),
                    "baseline_score": baseline_score,
                    "quality_score": combined_score,
                }
            ).to_parquet(ARTIFACTS / "tables" / "third_pass_may_quality_predictions.parquet", index=False)
        del train, validation

    april_delta = float(output["windows"]["april"]["combined_minus_baseline"]["pr_auc"])  # type: ignore[index]
    may_delta = float(output["windows"]["may"]["combined_minus_baseline"]["pr_auc"])  # type: ignore[index]
    keep = may_delta >= 0.002 and april_delta >= -0.0005
    output["pre_june_decision"] = {
        "status": "KEEP" if keep else "DROP",
        "rule": "KEEP requires >=0.002 May PR-AUC gain and no material (>0.0005) April regression.",
        "april_pr_auc_delta": april_delta,
        "may_pr_auc_delta": may_delta,
    }

    # June is opened only after the pre-June decision above and cannot affect it.
    june = _joined_frame(f"f.block_time>={JUNE_START}", all_numeric, categorical, quality=True)
    june_result: dict[str, object] = {}
    prediction_output = pd.DataFrame(
        {"token_address": june.token_address, "block_time": june.block_time, "label": june.label.astype("uint8")}
    )
    for name, bundle in may_models.items():
        score = predict(bundle, june)
        june_result[name] = {
            **metrics(june.label.to_numpy(dtype=np.uint8), score, may_thresholds[name]),
            "top_k": _rank_summary(june.label.to_numpy(dtype=np.uint8), score),
        }
        prediction_output[f"{name}_score"] = score
        prediction_output[f"{name}_selected"] = (score >= may_thresholds[name]).astype("uint8")
    prediction_output.to_parquet(
        ARTIFACTS / "tables" / "third_pass_june_quality_predictions.parquet", index=False
    )
    output["june_reporting_only"] = june_result
    pd.DataFrame(daily_stability_rows).to_csv(QUALITY_DAILY_STABILITY, index=False)
    QUALITY_AUDIT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def run_developer_sell_outcome_experiment(force: bool = False) -> dict[str, object]:
    """Test whether point-in-time sell behavior of prior launches adds quality signal."""
    ensure_output_dirs()
    build_dev_sell_features(force=force)
    temporal_audit = audit_dev_sell_features()
    baseline_numeric, categorical = _available_features(FEATURE_STORE)
    claim_numeric = baseline_numeric + HISTORICAL_OUTCOME_FEATURES
    sell_numeric = baseline_numeric + DEVELOPER_SELL_FEATURES
    all_numeric = claim_numeric + DEVELOPER_SELL_FEATURES
    stride = 2
    windows = {
        "april": {
            "train": f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{APRIL_START} "
            f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
            "validation": f"f.block_time>={APRIL_START} AND f.block_time<{MAY_START}",
        },
        "may": {
            "train": f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{MAY_START} "
            f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
            "validation": f"f.block_time>={MAY_START} AND f.block_time<{JUNE_START}",
        },
    }
    output: dict[str, object] = {
        "hypothesis": "Whether and how quickly a deployer sold its own prior Pump launches adds point-in-time signal beyond activity and creator-fee quality.",
        "semantics": "A developer sell is the deployer wallet's first sell event for a token it previously launched; it becomes visible only at the sell timestamp.",
        "temporal_audit": temporal_audit,
        "windows": {},
    }
    may_models: dict[str, ModelBundle] = {}
    may_thresholds: dict[str, float] = {}
    for period, bounds in windows.items():
        train = _joined_frame(
            bounds["train"], all_numeric, categorical, quality=True, dev_sell=True
        )
        validation = _joined_frame(
            bounds["validation"], all_numeric, categorical, quality=True, dev_sell=True
        )
        models = {
            "baseline": fit_lightgbm(train, baseline_numeric, categorical, stride),
            "creator_fee_quality": fit_lightgbm(train, claim_numeric, categorical, stride),
            "developer_sell_only": fit_lightgbm(train, DEVELOPER_SELL_FEATURES, [], stride),
            "baseline_plus_developer_sell": fit_lightgbm(
                train, sell_numeric, categorical, stride
            ),
            "creator_fee_plus_developer_sell": fit_lightgbm(
                train, all_numeric, categorical, stride
            ),
        }
        summaries: dict[str, dict[str, object]] = {}
        scores: dict[str, np.ndarray] = {}
        thresholds: dict[str, float] = {}
        for name, model in models.items():
            summary, score, threshold = _model_summary(model, validation)
            summaries[name] = summary
            scores[name] = score
            thresholds[name] = threshold
        incremental_daily, _ = _daily_pr_auc_stability(
            period,
            validation,
            scores["creator_fee_quality"],
            scores["creator_fee_plus_developer_sell"],
        )
        output["windows"][period] = {  # type: ignore[index]
            **summaries,
            "developer_sell_increment_over_creator_fee": {
                key: float(
                    summaries["creator_fee_plus_developer_sell"][key]
                    - summaries["creator_fee_quality"][key]
                )
                for key in ("pr_auc", "precision", "recall", "f1")
            },
            "developer_sell_group_permutation_pr_auc_drop": _group_permutation_drop(
                models["creator_fee_plus_developer_sell"],
                validation,
                DEVELOPER_SELL_FEATURES,
            ),
            "developer_sell_feature_gain_top15": _gain_importance(
                models["creator_fee_plus_developer_sell"], DEVELOPER_SELL_FEATURES
            ),
            "incremental_daily_pr_auc_stability": incremental_daily,
        }
        if period == "may":
            may_models = models
            may_thresholds = thresholds
        del train, validation

    april_delta = float(
        output["windows"]["april"]["developer_sell_increment_over_creator_fee"][
            "pr_auc"
        ]
    )  # type: ignore[index]
    may_delta = float(
        output["windows"]["may"]["developer_sell_increment_over_creator_fee"][
            "pr_auc"
        ]
    )  # type: ignore[index]
    keep = may_delta >= 0.002 and april_delta >= -0.0005
    output["pre_june_decision"] = {
        "status": "KEEP" if keep else "DROP",
        "rule": "Add developer-sell features only for >=0.002 May PR-AUC lift over creator-fee quality and no material (>0.0005) April regression.",
        "april_incremental_pr_auc": april_delta,
        "may_incremental_pr_auc": may_delta,
    }
    if keep:
        final_bundle = may_models["creator_fee_plus_developer_sell"]
        QUALITY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_bundle.model, QUALITY_MODEL_DIR / "final_model.joblib")
        (QUALITY_MODEL_DIR / "model_features.json").write_text(
            json.dumps(
                {
                    "numeric_features": final_bundle.numeric_features,
                    "categorical_features": final_bundle.categorical_features,
                    "transformed_feature_names": final_bundle.feature_names_out,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (QUALITY_MODEL_DIR / "operating_point.json").write_text(
            json.dumps(
                {
                    "model": "historical_outcome_augmented_active_era_lightgbm",
                    "feature_families": ["creator_fee_quality", "developer_sell_behavior"],
                    "selection_policy": "fixed score threshold maximizing F1 on May validation",
                    "threshold": may_thresholds["creator_fee_plus_developer_sell"],
                    "selected_pre_june_only": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    june = _joined_frame(
        f"f.block_time>={JUNE_START}",
        all_numeric,
        categorical,
        quality=True,
        dev_sell=True,
    )
    y = june.label.to_numpy(dtype=np.uint8)
    output["june_reporting_only"] = {
        name: {
            **metrics(y, predict(model, june), may_thresholds[name]),
            "top_k": _rank_summary(y, predict(model, june)),
        }
        for name, model in may_models.items()
    }
    DEV_SELL_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def _selected_historical_features() -> tuple[list[str], bool]:
    if DEV_SELL_RESULTS.exists():
        result = json.loads(DEV_SELL_RESULTS.read_text())
        if result.get("pre_june_decision", {}).get("status") == "KEEP":
            return HISTORICAL_OUTCOME_FEATURES + DEVELOPER_SELL_FEATURES, True
    return HISTORICAL_OUTCOME_FEATURES, False


def fit_time_bucket_ranker(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    stride: int,
) -> ModelBundle:
    """Fit one ranking formulation: deployments compete within a six-hour session."""
    ordered = train.assign(
        _query_bucket=train.block_time // RANKING_BUCKET_SECONDS
    ).sort_values(
        ["_query_bucket", "block_time", "token_address"]
    )
    groups = ordered.groupby("_query_bucket", sort=False).size().to_numpy()
    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale=False)),
            (
                "model",
                LGBMRanker(
                    objective="lambdarank",
                    n_estimators=500,
                    learning_rate=0.045,
                    num_leaves=31,
                    max_depth=7,
                    min_child_samples=120,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.2,
                    reg_lambda=2.0,
                    max_bin=127,
                    label_gain=[0, 1],
                    n_jobs=20,
                    random_state=20260811,
                    verbosity=-1,
                ),
            ),
        ]
    )
    weights = np.where(ordered.label.to_numpy() == 1, 1.0, float(stride))
    pipeline.fit(
        ordered[numeric + categorical],
        ordered.label,
        model__group=groups,
        model__sample_weight=weights,
    )
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    return ModelBundle(pipeline, numeric, categorical, names, "six_hour_lambdarank")


def _selection_metrics(y: np.ndarray, score: np.ndarray, selected: np.ndarray) -> dict[str, object]:
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, score))
    precision = float(precision_score(y, selected, zero_division=0))
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lift_over_prevalence": pr_auc / prevalence if prevalence else 0.0,
        "precision": precision,
        "precision_lift_over_prevalence": precision / prevalence if prevalence else 0.0,
        "recall": float(recall_score(y, selected, zero_division=0)),
        "f1": float(f1_score(y, selected, zero_division=0)),
        "predicted_entries": int(selected.sum()),
        "predicted_entry_rate": float(selected.mean()),
        "true_positives": int((selected & (y == 1)).sum()),
        "top_k": _rank_summary(y, score),
    }


def _bucket_rate_selection(
    validation: pd.DataFrame,
    score: np.ndarray,
    rate: float,
) -> np.ndarray:
    selected = np.zeros(len(validation), dtype=bool)
    buckets = validation.block_time.to_numpy(dtype=np.int64) // RANKING_BUCKET_SECONDS
    for bucket in np.unique(buckets):
        indices = np.flatnonzero(buckets == bucket)
        k = min(len(indices), max(1, int(round(rate * len(indices)))))
        chosen = indices[np.argpartition(score[indices], -k)[-k:]]
        selected[chosen] = True
    return selected


def fit_chronological_hard_negative_model(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    stride: int,
    mining_cutoff: int,
) -> tuple[ModelBundle, dict[str, object]]:
    """Mine near misses only in a later training slice scored by earlier data."""
    ordered = train.reset_index(drop=True)
    mining_train = ordered[ordered.block_time < mining_cutoff]
    mining_pool_mask = (ordered.block_time >= mining_cutoff) & ordered.label.eq(0)
    if mining_train.label.sum() < 100 or mining_pool_mask.sum() == 0:
        raise RuntimeError("hard-negative mining window lacks adequate examples")
    miner = fit_lightgbm(mining_train, numeric, categorical, stride)
    pool_positions = np.flatnonzero(mining_pool_mask.to_numpy())
    pool_scores = predict(miner, ordered.iloc[pool_positions])
    late_positives = int(
        ordered.loc[ordered.block_time >= mining_cutoff, "label"].sum()
    )
    hard_count = min(len(pool_positions), max(1, 5 * late_positives))
    hard_local = np.argpartition(pool_scores, -hard_count)[-hard_count:]
    hard_positions = pool_positions[hard_local]
    extra_weight = np.ones(len(ordered), dtype=float)
    extra_weight[hard_positions] = 3.0
    model = fit_lightgbm(
        ordered,
        numeric,
        categorical,
        stride,
        extra_sample_weight=extra_weight,
    )
    return model, {
        "mining_cutoff_unix": mining_cutoff,
        "miner_rows": int(len(mining_train)),
        "miner_positives": int(mining_train.label.sum()),
        "scored_later_training_negatives": int(len(pool_positions)),
        "later_training_positives": late_positives,
        "hard_negative_count": hard_count,
        "hard_negative_weight_multiplier": 3.0,
        "selection_rule": "top five scored later-slice negatives per later-slice positive",
    }


def _missing_history_mixture(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    baseline: ModelBundle,
    baseline_score: np.ndarray,
    baseline_threshold: float,
    no_activity_numeric: list[str],
    no_activity_categorical: list[str],
    stride: int,
) -> tuple[dict[str, object], ModelBundle, np.ndarray, float]:
    missing_train = train[train.history_missing.eq(1)].copy()
    missing_validation = validation[validation.history_missing.eq(1)].copy()
    submodel = fit_lightgbm(
        missing_train, no_activity_numeric, no_activity_categorical, stride
    )
    subscore = predict(submodel, missing_validation)
    subthreshold = _thresholds(
        missing_validation.label.to_numpy(dtype=np.uint8), subscore
    )["max_f1"]
    mixed_score = baseline_score.copy()
    missing_mask = validation.history_missing.eq(1).to_numpy()
    mixed_score[missing_mask] = subscore
    selected = (baseline_score >= baseline_threshold) & ~missing_mask
    selected[missing_mask] = subscore >= subthreshold
    result = _selection_metrics(
        validation.label.to_numpy(dtype=np.uint8), mixed_score, selected
    )
    result["missing_path"] = {
        "train_rows": int(len(missing_train)),
        "train_positives": int(missing_train.label.sum()),
        "validation_rows": int(len(missing_validation)),
        "validation_positives": int(missing_validation.label.sum()),
        "threshold": subthreshold,
        "selected": int((subscore >= subthreshold).sum()),
        "true_positives": int(
            ((subscore >= subthreshold) & missing_validation.label.eq(1).to_numpy()).sum()
        ),
        "pr_auc": float(
            average_precision_score(missing_validation.label.to_numpy(), subscore)
        ),
    }
    return result, submodel, mixed_score, subthreshold


def run_ranking_hard_negative_experiment() -> dict[str, object]:
    """Compare ranking, rate control, hard negatives, and an evidenced sub-policy."""
    ensure_output_dirs()
    numeric, categorical = _available_features(FEATURE_STORE)
    no_activity_groups = [
        group
        for group in ("timing", "deployment_history", "metadata", "signed_message")
    ]
    no_activity_numeric, no_activity_categorical = _available_features(
        FEATURE_STORE, no_activity_groups
    )
    stride = 2
    windows = {
        "april": {
            "train": f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{APRIL_START} "
            f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
            "validation": f"f.block_time>={APRIL_START} AND f.block_time<{MAY_START}",
            "mining_cutoff": 1_774_310_400,  # 2026-03-24 UTC
        },
        "may": {
            "train": f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{MAY_START} "
            f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
            "validation": f"f.block_time>={MAY_START} AND f.block_time<{JUNE_START}",
            "mining_cutoff": APRIL_START,
        },
    }
    output: dict[str, object] = {
        "ranking_query_semantics": "Six-hour UTC bucket: deployments compete with candidates arriving in the same operational session. Daily groups exceeded LightGBM's 10,000-row query limit on the sampled population.",
        "hard_negative_leakage_policy": "Only a later training slice is mined, using a model fit on an earlier training slice; validation labels never identify hard negatives.",
        "windows": {},
    }
    may_artifacts: dict[str, object] = {}
    for window, bounds in windows.items():
        train = _joined_frame(bounds["train"], numeric, categorical)
        validation = _joined_frame(bounds["validation"], numeric, categorical)
        baseline = fit_lightgbm(train, numeric, categorical, stride)
        baseline_result, baseline_score, baseline_threshold = _model_summary(
            baseline, validation
        )

        ranker = fit_time_bucket_ranker(train, numeric, categorical, stride)
        ranker_result, ranker_score, ranker_threshold = _model_summary(
            ranker, validation
        )
        effective_population = int(train.label.sum()) + stride * int(
            train.label.eq(0).sum()
        )
        train_rate = float(train.label.sum() / effective_population)
        rate_selected = _bucket_rate_selection(validation, ranker_score, train_rate)
        rate_result = _selection_metrics(
            validation.label.to_numpy(dtype=np.uint8), ranker_score, rate_selected
        )
        rate_result["training_population_positive_rate"] = train_rate

        hard_model, mining = fit_chronological_hard_negative_model(
            train,
            numeric,
            categorical,
            stride,
            int(bounds["mining_cutoff"]),
        )
        hard_result, hard_score, hard_threshold = _model_summary(
            hard_model, validation
        )

        mixture_result, missing_model, mixture_score, missing_threshold = (
            _missing_history_mixture(
                train,
                validation,
                baseline,
                baseline_score,
                baseline_threshold,
                no_activity_numeric,
                no_activity_categorical,
                stride,
            )
        )
        output["windows"][window] = {  # type: ignore[index]
            "baseline": baseline_result,
            "time_bucket_lambdarank_thresholded": ranker_result,
            "time_bucket_lambdarank_training_rate_policy": rate_result,
            "chronological_hard_negative": hard_result,
            "hard_negative_mining": mining,
            "missing_history_policy_mixture": mixture_result,
            "pr_auc_deltas_vs_baseline": {
                "time_bucket_lambdarank": float(ranker_result["pr_auc"] - baseline_result["pr_auc"]),
                "hard_negative": float(hard_result["pr_auc"] - baseline_result["pr_auc"]),
                "missing_history_mixture": float(mixture_result["pr_auc"] - baseline_result["pr_auc"]),
            },
            "f1_deltas_vs_baseline": {
                "time_bucket_lambdarank": float(ranker_result["f1"] - baseline_result["f1"]),
                "time_bucket_rate_policy": float(rate_result["f1"] - baseline_result["f1"]),
                "hard_negative": float(hard_result["f1"] - baseline_result["f1"]),
                "missing_history_mixture": float(mixture_result["f1"] - baseline_result["f1"]),
            },
        }
        if window == "may":
            may_artifacts = {
                "validation": validation,
                "models": {
                    "baseline": baseline,
                    "ranker": ranker,
                    "hard_negative": hard_model,
                    "missing_history": missing_model,
                },
                "thresholds": {
                    "baseline": baseline_threshold,
                    "ranker": ranker_threshold,
                    "hard_negative": hard_threshold,
                    "missing_history": missing_threshold,
                },
            }
            pd.DataFrame(
                {
                    "token_address": validation.token_address,
                    "label": validation.label.astype("uint8"),
                    "baseline_score": baseline_score,
                    "ranker_score": ranker_score,
                    "hard_negative_score": hard_score,
                    "mixture_score": mixture_score,
                }
            ).to_parquet(
                ARTIFACTS / "tables" / "third_pass_may_policy_predictions.parquet",
                index=False,
            )

    decisions: dict[str, object] = {}
    for method in ("time_bucket_lambdarank", "hard_negative", "missing_history_mixture"):
        april_delta = float(output["windows"]["april"]["pr_auc_deltas_vs_baseline"][method])  # type: ignore[index]
        may_delta = float(output["windows"]["may"]["pr_auc_deltas_vs_baseline"][method])  # type: ignore[index]
        decisions[method] = {
            "april_pr_auc_delta": april_delta,
            "may_pr_auc_delta": may_delta,
            "status": "KEEP" if april_delta > 0.001 and may_delta > 0.001 else "DROP",
        }
    output["pre_june_decisions"] = decisions

    # No June method can change the decisions above.
    june = _joined_frame(f"f.block_time>={JUNE_START}", numeric, categorical)
    y_june = june.label.to_numpy(dtype=np.uint8)
    june_results: dict[str, object] = {}
    models = may_artifacts["models"]  # type: ignore[index]
    thresholds = may_artifacts["thresholds"]  # type: ignore[index]
    for name in ("baseline", "ranker", "hard_negative"):
        score = predict(models[name], june)
        june_results[name] = {
            **metrics(y_june, score, thresholds[name]),
            "top_k": _rank_summary(y_june, score),
        }
    missing_mask = june.history_missing.eq(1).to_numpy()
    baseline_june_score = predict(models["baseline"], june)
    missing_june_score = predict(
        models["missing_history"], june[missing_mask]
    )
    mixed_score = baseline_june_score.copy()
    mixed_score[missing_mask] = missing_june_score
    mixed_selected = (baseline_june_score >= thresholds["baseline"]) & ~missing_mask
    mixed_selected[missing_mask] = missing_june_score >= thresholds["missing_history"]
    june_results["missing_history_mixture"] = _selection_metrics(
        y_june, mixed_score, mixed_selected
    )
    output["june_reporting_only"] = june_results
    POLICY_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def _week_start(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, unit="s", utc=True).dt.floor("D")
    return (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.strftime(
        "%Y-%m-%d"
    )


def run_weekly_regime_analysis() -> dict[str, object]:
    """Describe active-era drift with models trained only on earlier months."""
    ensure_output_dirs()
    numeric, categorical = _available_features(FEATURE_STORE)
    stride = 2
    march_train = _joined_frame(
        f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{APRIL_START} "
        f"AND (f.label=1 OR hash(f.token_address)%{stride}=0)",
        numeric,
        categorical,
    )
    april_model = fit_lightgbm(march_train, numeric, categorical, stride)
    feature_meta = json.loads((MODEL_DIR / "model_features.json").read_text())
    final_model = ModelBundle(
        model=joblib.load(MODEL_DIR / "final_model.joblib"),
        numeric_features=feature_meta["numeric_features"],
        categorical_features=feature_meta["categorical_features"],
        feature_names_out=feature_meta["transformed_feature_names"],
        name="tracked_active_era_lightgbm",
    )
    final_threshold = float(
        json.loads((MODEL_DIR / "operating_point.json").read_text())["threshold"]
    )
    diagnostic_features = [
        "seconds_since_prior_deploy",
        "hist_open_close_count",
        "hist_quote_sol_sum",
        "dev_buy_sol",
        "hist_cost_usd_sum",
        "observed_wallet_age_seconds",
        "prior_deploy_count_7d",
        "hist_tip_fee_sum",
        "prior_deploy_count_1d",
        "uri_length",
        "hist_claimed_launch_fraction",
        "hist_claim_fee_usd_sum",
        "hist_mature_7d_success_fraction",
    ]
    quality_diagnostic_features = [
        "hist_claimed_launch_fraction",
        "hist_claim_fee_usd_sum",
        "hist_mature_7d_success_fraction",
    ]
    periods = [
        ("april", APRIL_START, MAY_START, april_model, None),
        ("may", MAY_START, JUNE_START, final_model, final_threshold),
        ("june", JUNE_START, 1_782_864_000, final_model, final_threshold),
    ]
    scored_parts: list[pd.DataFrame] = []
    monthly_thresholds: dict[str, float] = {}
    for name, start, end, bundle, frozen_threshold in periods:
        frame = _joined_frame(
            f"f.block_time>={start} AND f.block_time<{end}",
            numeric + quality_diagnostic_features,
            categorical,
            quality=True,
        )
        score = predict(bundle, frame)
        threshold = (
            float(frozen_threshold)
            if frozen_threshold is not None
            else _thresholds(frame.label.to_numpy(dtype=np.uint8), score)["max_f1"]
        )
        monthly_thresholds[name] = threshold
        keep = frame[
            [
                "token_address",
                "tx_signer",
                "block_time",
                "label",
                "history_missing",
                *diagnostic_features,
            ]
        ].copy()
        keep["score"] = score
        keep["selected"] = score >= threshold
        keep["model_origin"] = (
            "trained March 12-31" if name == "april" else "trained March 12-April"
        )
        keep["month"] = name
        scored_parts.append(keep)
        del frame
    scored = pd.concat(scored_parts, ignore_index=True)
    scored["week_start_utc"] = _week_start(scored.block_time)

    weekly_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    for (period, week), frame in scored.groupby(["month", "week_start_utc"], sort=True):
        y = frame.label.to_numpy(dtype=np.uint8)
        score = frame.score.to_numpy()
        selected = frame.selected.to_numpy(dtype=bool)
        positive = frame[frame.label.eq(1)]
        signer_counts = positive.tx_signer.value_counts()
        week_metrics = _selection_metrics(y, score, selected)
        weekly_optimal_threshold = _thresholds(y, score)["max_f1"]
        weekly_rows.append(
            {
                "week_start_utc": week,
                "period": period,
                "model_origin": frame.model_origin.iloc[0],
                "rows": int(len(frame)),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "target_buys_per_100k_candidates": float(100_000 * y.mean()),
                "positive_signers": int(positive.tx_signer.nunique()),
                "positive_top10_signer_share": float(
                    signer_counts.head(10).sum() / len(positive)
                )
                if len(positive)
                else 0.0,
                "positive_history_missing_share": float(
                    positive.history_missing.mean()
                )
                if len(positive)
                else 0.0,
                "positive_dev_buy_zero_share": float(positive.dev_buy_sol.eq(0).mean())
                if len(positive)
                else 0.0,
                "positive_dev_buy_gt10_share": float(positive.dev_buy_sol.gt(10).mean())
                if len(positive)
                else 0.0,
                "score_mean": float(score.mean()),
                "score_p50": float(np.quantile(score, 0.50)),
                "score_p90": float(np.quantile(score, 0.90)),
                "score_p99": float(np.quantile(score, 0.99)),
                "actual_to_mean_score_ratio": float(y.mean() / score.mean())
                if score.mean() > 0
                else None,
                "frozen_threshold": float(
                    monthly_thresholds[frame.month.iloc[0]]
                ),
                "diagnostic_weekly_f1_threshold": weekly_optimal_threshold,
                "pr_auc": week_metrics["pr_auc"],
                "precision": week_metrics["precision"],
                "recall": week_metrics["recall"],
                "f1": week_metrics["f1"],
                "selected": week_metrics["predicted_entries"],
            }
        )
        for feature in diagnostic_features:
            full = pd.to_numeric(frame[feature], errors="coerce")
            positives = full[frame.label.eq(1)]
            negatives = full[frame.label.eq(0)]
            q25, q75 = full.quantile([0.25, 0.75])
            scale = float(q75 - q25)
            positive_median = float(positives.median()) if positives.notna().any() else None
            negative_median = float(negatives.median()) if negatives.notna().any() else None
            standardized_gap = (
                (positive_median - negative_median) / scale
                if positive_median is not None
                and negative_median is not None
                and scale > 0
                else None
            )
            effect_rows.append(
                {
                    "week_start_utc": week,
                    "period": period,
                    "feature": feature,
                    "positive_median": positive_median,
                    "negative_median": negative_median,
                    "population_iqr": scale,
                    "positive_minus_negative_iqr": standardized_gap,
                }
            )
    weekly = pd.DataFrame(weekly_rows)
    effects = pd.DataFrame(effect_rows)
    effects["absolute_gap"] = effects.positive_minus_negative_iqr.abs()
    effects["within_week_effect_rank"] = effects.groupby(["period", "week_start_utc"])[
        "absolute_gap"
    ].rank(method="min", ascending=False)
    weekly.to_csv(WEEKLY_TABLE, index=False)
    effects.drop(columns="absolute_gap").to_csv(WEEKLY_EFFECTS, index=False)

    april_weeks = weekly[weekly.period.eq("april")]
    may_weeks = weekly[weekly.period.eq("may")]
    june_weeks = weekly[weekly.period.eq("june")]
    output: dict[str, object] = {
        "scope": "Weekly April-June diagnostics; March 12-31 trains the April model, and March 12-April trains the frozen May/June model.",
        "threshold_policy": {
            "april": "full-April diagnostic threshold for the March-trained model",
            "may_and_june": "tracked May-selected threshold; June never recalibrated",
            "weekly_optimal_threshold": "post-hoc drift diagnostic only; never used as a policy",
        },
        "monthly_thresholds": monthly_thresholds,
        "window_ranges": {
            "april_pr_auc": [
                float(april_weeks.pr_auc.min()),
                float(april_weeks.pr_auc.max()),
            ],
            "may_pr_auc": [float(may_weeks.pr_auc.min()), float(may_weeks.pr_auc.max())],
            "june_pr_auc": [
                float(june_weeks.pr_auc.min()),
                float(june_weeks.pr_auc.max()),
            ],
            "may_positive_rate_per_100k": [
                float(may_weeks.target_buys_per_100k_candidates.min()),
                float(may_weeks.target_buys_per_100k_candidates.max()),
            ],
            "june_positive_rate_per_100k": [
                float(june_weeks.target_buys_per_100k_candidates.min()),
                float(june_weeks.target_buys_per_100k_candidates.max()),
            ],
        },
        "metrics_path": "submission/tables/weekly_regime_metrics.csv",
        "feature_effects_path": "submission/tables/weekly_feature_effects.csv",
    }
    WEEKLY_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def _economic_frame(
    start: int,
    end: int,
    numeric: list[str],
    categorical: list[str],
    *,
    training_stride: int | None = None,
    dev_sell: bool = False,
) -> pd.DataFrame:
    build_claim_outcomes()
    predicate = f"f.block_time>={start} AND f.block_time<{end}"
    if training_stride is not None:
        predicate += (
            f" AND (hash(f.token_address)%{training_stride}=0 OR f.token_address IN ("
            f"SELECT token_address FROM read_parquet('{_sql(CLAIM_OUTCOMES)}') "
            "WHERE creator_fee_claim_events_7d>0))"
        )
    frame = _joined_frame(
        predicate,
        numeric,
        categorical,
        quality=True,
        dev_sell=dev_sell,
        outcome=True,
    )
    frame["target_label"] = frame.label.astype("uint8")
    frame["label"] = frame.creator_fee_claim_events_7d.gt(0).astype("uint8")
    return frame


def _strategy_selection_summary(
    frame: pd.DataFrame,
    selected: np.ndarray,
) -> dict[str, float | int]:
    selected_frame = frame.loc[selected]
    target = frame.target_label.to_numpy(dtype=np.uint8)
    claim = frame.label.to_numpy(dtype=np.uint8)
    return {
        "entries": int(selected.sum()),
        "target_overlap": int(target[selected].sum()),
        "precision_vs_target": float(target[selected].mean()) if selected.any() else 0.0,
        "recall_of_target": float(target[selected].sum() / target.sum()) if target.sum() else 0.0,
        "creator_fee_claim_7d_hit_rate": float(claim[selected].mean()) if selected.any() else 0.0,
        "creator_fee_usd_7d_total": float(selected_frame.creator_fee_usd_7d.sum()),
        "creator_fee_usd_7d_mean": float(selected_frame.creator_fee_usd_7d.mean())
        if selected.any()
        else 0.0,
        "creator_fee_usd_7d_p99_capped_total": float(
            selected_frame.creator_fee_usd_7d.clip(
                upper=selected_frame.creator_fee_usd_7d.quantile(0.99)
            ).sum()
        )
        if selected.any()
        else 0.0,
    }


def _two_stage_selection(
    bot_score: np.ndarray,
    economic_score: np.ndarray,
    entries: int,
) -> tuple[np.ndarray, np.ndarray]:
    if entries <= 0:
        return np.zeros(len(bot_score), dtype=bool), np.full(len(bot_score), -np.inf)
    gate_size = min(len(bot_score), 2 * entries)
    gate = np.argpartition(bot_score, -gate_size)[-gate_size:]
    bot_rank = pd.Series(bot_score[gate]).rank(method="average", pct=True).to_numpy()
    quality_rank = pd.Series(economic_score[gate]).rank(method="average", pct=True).to_numpy()
    combined_local = 0.5 * bot_rank + 0.5 * quality_rank
    chosen_local = np.argpartition(combined_local, -entries)[-entries:]
    selected = np.zeros(len(bot_score), dtype=bool)
    selected[gate[chosen_local]] = True
    combined_score = np.full(len(bot_score), -np.inf)
    combined_score[gate] = combined_local
    return selected, combined_score


def _frontier_rows(
    period: str,
    frame: pd.DataFrame,
    bot_score: np.ndarray,
    combined_score: np.ndarray,
    base_entries: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for multiplier in (0.25, 0.5, 1.0, 1.5, 2.0):
        entries = min(len(frame), max(1, int(round(base_entries * multiplier))))
        for policy, score in (("bot_score", bot_score), ("two_stage", combined_score)):
            finite = np.flatnonzero(np.isfinite(score))
            entries_used = min(entries, len(finite))
            chosen = finite[np.argpartition(score[finite], -entries_used)[-entries_used:]]
            selected = np.zeros(len(frame), dtype=bool)
            selected[chosen] = True
            rows.append(
                {
                    "period": period,
                    "policy": policy,
                    "trade_count_multiplier": multiplier,
                    **_strategy_selection_summary(frame, selected),
                }
            )
    return rows


def _build_custom_strategy_outcomes(predictions: pd.DataFrame) -> pd.DataFrame:
    con = _connection()
    con.register("predictions", predictions)
    result = con.execute(
        f"""
        WITH wanted AS (
          SELECT * FROM predictions
          WHERE baseline_selected=1 OR quality_selected=1 OR two_stage_selected=1
             OR selective_two_stage_selected=1 OR label=1
        ), trades AS MATERIALIZED (
          SELECT t.token_address,t.block_slot,t.tx_index,t.event_index,t.block_time,
                 t.price_sol,t.deploy_block_slot,t.deploy_tx_index
          FROM read_parquet('{_sql(JUNE_TRADES)}') t
          SEMI JOIN wanted w USING(token_address)
          WHERE t.price_sol>0
        ), policies(policy,tx_offset) AS (VALUES ('immediate',1),('offset_118',118)),
        candidates AS (
          SELECT w.*,p.policy,p.tx_offset,t.block_slot entry_slot,t.tx_index entry_tx_index,
                 t.event_index entry_event_index,t.block_time entry_time,
                 t.price_sol entry_price_sol,
                 row_number() OVER(PARTITION BY w.token_address,p.policy
                   ORDER BY t.block_slot,t.tx_index,t.event_index) rn
          FROM wanted w CROSS JOIN policies p JOIN trades t USING(token_address)
          WHERE (p.policy='immediate' AND (
                   t.block_slot>t.deploy_block_slot OR
                   (t.block_slot=t.deploy_block_slot AND t.tx_index>t.deploy_tx_index)))
             OR (p.policy='offset_118' AND t.block_slot=t.deploy_block_slot
                   AND t.tx_index>=t.deploy_tx_index+118)
        ), entries AS (SELECT * EXCLUDE(rn) FROM candidates WHERE rn=1),
        exit_candidates AS (
          SELECT e.token_address,e.policy,t.block_slot exit_slot,t.tx_index exit_tx_index,
                 t.event_index exit_event_index,t.block_time exit_time,t.price_sol exit_price_sol,
                 row_number() OVER(PARTITION BY e.token_address,e.policy
                   ORDER BY t.block_slot,t.tx_index,t.event_index) rn
          FROM entries e JOIN trades t USING(token_address)
          WHERE t.block_time>=e.entry_time+6
        ), exits AS (SELECT * EXCLUDE(rn) FROM exit_candidates WHERE rn=1)
        SELECT e.*,x.exit_slot,x.exit_tx_index,x.exit_event_index,x.exit_time,x.exit_price_sol,
               CASE WHEN x.exit_price_sol IS NULL THEN -1.0
                    ELSE x.exit_price_sol/e.entry_price_sol-1 END gross_roi,
               (x.exit_price_sol IS NULL)::UTINYINT forced_total_loss
        FROM entries e LEFT JOIN exits x USING(token_address,policy)
        """
    ).fetch_df()
    result.to_parquet(THIRD_STRATEGY_OUTCOMES, index=False)
    return result


def _disagreement_characterization(
    june_predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    immediate = outcomes[outcomes.policy.eq("immediate")].copy()
    stake, _, network_cost = _strategy_parameters()
    immediate["network_cost_adjusted_roi"] = immediate.gross_roi - network_cost / stake
    immediate["cohort"] = np.select(
        [
            immediate.label.eq(1) & immediate.baseline_selected.eq(1),
            immediate.label.eq(1) & immediate.baseline_selected.eq(0),
            immediate.label.eq(0)
            & immediate.baseline_selected.eq(1)
            & immediate.network_cost_adjusted_roi.gt(0),
            immediate.label.eq(0)
            & immediate.baseline_selected.eq(1)
            & immediate.network_cost_adjusted_roi.le(0),
        ],
        [
            "target_replica_overlap",
            "target_only",
            "replica_only_profitable",
            "replica_only_unprofitable",
        ],
        default="other",
    )
    feature_names = [
        "score",
        "dev_buy_sol",
        "seconds_since_prior_deploy",
        "hist_quote_sol_sum",
        "observed_wallet_age_seconds",
        "prior_deploy_count_7d",
        "hist_claimed_launch_count",
        "hist_mature_1d_success_fraction",
    ]
    con = _connection("24GB")
    con.register(
        "cohorts",
        immediate[["token_address", "cohort", "network_cost_adjusted_roi"]],
    )
    con.register("predictions", june_predictions)
    joined = con.execute(
        f"""
        SELECT c.*,p.bot_score AS score,f.dev_buy_sol,f.seconds_since_prior_deploy,
               f.hist_quote_sol_sum,f.observed_wallet_age_seconds,f.prior_deploy_count_7d,
               q.hist_claimed_launch_count,q.hist_mature_1d_success_fraction
        FROM cohorts c JOIN predictions p USING(token_address)
        JOIN read_parquet('{_sql(FEATURE_STORE)}') f USING(token_address)
        JOIN read_parquet('{_sql(QUALITY_FEATURES)}') q USING(token_address)
        """
    ).fetch_df()
    rows: list[dict[str, object]] = []
    for cohort, frame in joined[joined.cohort.ne("other")].groupby("cohort"):
        row: dict[str, object] = {
            "cohort": cohort,
            "tokens": int(len(frame)),
            "median_network_cost_adjusted_roi": float(frame.network_cost_adjusted_roi.median()),
            "mean_network_cost_adjusted_roi": float(frame.network_cost_adjusted_roi.mean()),
        }
        for feature in feature_names:
            row[f"median_{feature}"] = float(frame[feature].median())
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(DISAGREEMENT_COHORTS, index=False)
    return result


def run_profitable_disagreement_experiment() -> dict[str, object]:
    """Train Model B on a pre-June creator-fee proxy, then freeze June strategy."""
    ensure_output_dirs()
    build_quality_features()
    build_claim_outcomes()
    baseline_numeric, categorical = _available_features(FEATURE_STORE)
    historical_features, use_dev_sell = _selected_historical_features()
    numeric = baseline_numeric + historical_features
    economic_stride = 5
    configurations = {
        "april": {
            "economic_train_end": APRIL_START - 7 * DAY,
            "validation_start": APRIL_START,
            "validation_end": MAY_START - 7 * DAY,
            "bot_train_end": APRIL_START,
        },
        "may": {
            "economic_train_end": MAY_START - 7 * DAY,
            "validation_start": MAY_START,
            "validation_end": JUNE_START - 7 * DAY,
            "bot_train_end": MAY_START,
        },
    }
    output: dict[str, object] = {
        "control_model_a": "tracked target-bot buy probability without historical launch outcomes",
        "model_a": "pre-June-selected historical-outcome-augmented target-bot buy probability",
        "model_a_historical_feature_families": [
            "creator_fee_quality",
            *(["developer_sell_behavior"] if use_dev_sell else []),
        ],
        "model_b": "probability the creator claims a Pump creator fee within seven days",
        "model_b_guardrail": "Creator-fee claim is a sparse economic-quality proxy, not ROI. Each training candidate is at least seven days before its evaluation boundary.",
        "combination": "Gate to the top 2x Model-A count, average within-gate percentile ranks from Models A and B, then keep the same count as Model A.",
        "windows": {},
    }
    frontier_rows: list[dict[str, object]] = []
    for period, cfg in configurations.items():
        economic_train = _economic_frame(
            ACTIVE_ERA_START,
            int(cfg["economic_train_end"]),
            numeric,
            categorical,
            training_stride=economic_stride,
            dev_sell=use_dev_sell,
        )
        validation = _economic_frame(
            int(cfg["validation_start"]),
            int(cfg["validation_end"]),
            numeric,
            categorical,
            dev_sell=use_dev_sell,
        )
        economic_model = fit_lightgbm(
            economic_train, numeric, categorical, economic_stride
        )
        economic_score = predict(economic_model, validation)
        economic_metrics = metrics(
            validation.label.to_numpy(dtype=np.uint8),
            economic_score,
            _thresholds(validation.label.to_numpy(dtype=np.uint8), economic_score)[
                "max_f1"
            ],
        )

        control_bot_train = _joined_frame(
            f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{int(cfg['bot_train_end'])} "
            "AND (f.label=1 OR hash(f.token_address)%2=0)",
            baseline_numeric,
            categorical,
        )
        quality_bot_train = _joined_frame(
            f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{int(cfg['bot_train_end'])} "
            "AND (f.label=1 OR hash(f.token_address)%2=0)",
            numeric,
            categorical,
            quality=True,
            dev_sell=use_dev_sell,
        )
        control_bot_model = fit_lightgbm(
            control_bot_train, baseline_numeric, categorical, 2
        )
        quality_bot_model = fit_lightgbm(
            quality_bot_train, numeric, categorical, 2
        )
        control_bot_score = predict(control_bot_model, validation)
        control_bot_threshold = _thresholds(
            validation.target_label.to_numpy(dtype=np.uint8), control_bot_score
        )["max_f1"]
        control_bot_selected = control_bot_score >= control_bot_threshold
        bot_score = predict(quality_bot_model, validation)
        bot_threshold = _thresholds(
            validation.target_label.to_numpy(dtype=np.uint8), bot_score
        )["max_f1"]
        bot_selected = bot_score >= bot_threshold
        two_stage_selected, combined_score = _two_stage_selection(
            bot_score, economic_score, int(bot_selected.sum())
        )
        control_strategy = _strategy_selection_summary(
            validation, control_bot_selected
        )
        quality_strategy = _strategy_selection_summary(validation, bot_selected)
        two_stage_strategy = _strategy_selection_summary(
            validation, two_stage_selected
        )
        output["windows"][period] = {  # type: ignore[index]
            "economic_train_rows": int(len(economic_train)),
            "economic_train_positives": int(economic_train.label.sum()),
            "validation_rows": int(len(validation)),
            "validation_economic_positives": int(validation.label.sum()),
            "model_b_metrics": economic_metrics,
            "control_model_a_strategy": control_strategy,
            "quality_model_a_strategy": quality_strategy,
            "two_stage_strategy": two_stage_strategy,
            "quality_model_a_minus_control": {
                "precision_vs_target": float(
                    quality_strategy["precision_vs_target"]
                    - control_strategy["precision_vs_target"]
                ),
                "recall_of_target": float(
                    quality_strategy["recall_of_target"]
                    - control_strategy["recall_of_target"]
                ),
            },
            "two_stage_minus_quality_model_a": {
                "creator_fee_claim_hit_rate": float(
                    two_stage_strategy["creator_fee_claim_7d_hit_rate"]
                    - quality_strategy["creator_fee_claim_7d_hit_rate"]
                ),
                "creator_fee_usd_7d_p99_capped_total": float(
                    two_stage_strategy["creator_fee_usd_7d_p99_capped_total"]
                    - quality_strategy["creator_fee_usd_7d_p99_capped_total"]
                ),
                "precision_vs_target": float(
                    two_stage_strategy["precision_vs_target"]
                    - quality_strategy["precision_vs_target"]
                ),
            },
        }
        frontier_rows.extend(
            _frontier_rows(
                period,
                validation,
                bot_score,
                combined_score,
                int(bot_selected.sum()),
            )
        )
    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(STRATEGY_FRONTIER, index=False)
    april_gain = float(
        output["windows"]["april"]["two_stage_minus_quality_model_a"][
            "creator_fee_claim_hit_rate"
        ]
    )  # type: ignore[index]
    may_gain = float(
        output["windows"]["may"]["two_stage_minus_quality_model_a"][
            "creator_fee_claim_hit_rate"
        ]
    )  # type: ignore[index]
    keep = april_gain > 0 and may_gain > 0
    output["pre_june_decision"] = {
        "status": "KEEP" if keep else "DROP",
        "april_claim_hit_rate_gain": april_gain,
        "may_claim_hit_rate_gain": may_gain,
        "rule": "KEEP only if equal-count two-stage selection improves the seven-day creator-fee hit rate in both April and May.",
    }
    quarter = frontier[frontier.trade_count_multiplier.eq(0.25)].set_index(
        ["period", "policy"]
    )
    quarter_better = all(
        quarter.loc[(period, "two_stage"), "creator_fee_claim_7d_hit_rate"]
        > quarter.loc[(period, "bot_score"), "creator_fee_claim_7d_hit_rate"]
        for period in ("april", "may")
    )
    output["pre_june_strategy_operating_point"] = {
        "status": "KEEP" if quarter_better else "DROP",
        "trade_count_multiplier": 0.25,
        "selection_basis": "Smallest tested pre-June operating point; creator-fee hit rate exceeds Model A in both April and May.",
        "april_two_stage_claim_hit_rate": float(
            quarter.loc[("april", "two_stage"), "creator_fee_claim_7d_hit_rate"]
        ),
        "may_two_stage_claim_hit_rate": float(
            quarter.loc[("may", "two_stage"), "creator_fee_claim_7d_hit_rate"]
        ),
    }

    # Final Model B uses only candidates whose full seven-day outcome is known by June 1.
    final_economic_train = _economic_frame(
        ACTIVE_ERA_START,
        JUNE_START - 7 * DAY,
        numeric,
        categorical,
        training_stride=economic_stride,
        dev_sell=use_dev_sell,
    )
    final_economic_model = fit_lightgbm(
        final_economic_train, numeric, categorical, economic_stride
    )
    june = _joined_frame(
        f"f.block_time>={JUNE_START}",
        numeric,
        categorical,
        quality=True,
        dev_sell=use_dev_sell,
    )
    feature_meta = json.loads((MODEL_DIR / "model_features.json").read_text())
    control_bot_bundle = ModelBundle(
        model=joblib.load(MODEL_DIR / "final_model.joblib"),
        numeric_features=feature_meta["numeric_features"],
        categorical_features=feature_meta["categorical_features"],
        feature_names_out=feature_meta["transformed_feature_names"],
        name="tracked_active_era_lightgbm",
    )
    if not (QUALITY_MODEL_DIR / "final_model.joblib").exists():
        run_historical_outcome_experiment()
    quality_meta = json.loads((QUALITY_MODEL_DIR / "model_features.json").read_text())
    quality_bot_bundle = ModelBundle(
        model=joblib.load(QUALITY_MODEL_DIR / "final_model.joblib"),
        numeric_features=quality_meta["numeric_features"],
        categorical_features=quality_meta["categorical_features"],
        feature_names_out=quality_meta["transformed_feature_names"],
        name="quality_augmented_active_era_lightgbm",
    )
    control_bot_score = predict(control_bot_bundle, june)
    quality_bot_score = predict(quality_bot_bundle, june)
    economic_score = predict(final_economic_model, june)
    control_bot_threshold = float(
        json.loads((MODEL_DIR / "operating_point.json").read_text())["threshold"]
    )
    quality_bot_threshold = float(
        json.loads((QUALITY_MODEL_DIR / "operating_point.json").read_text())[
            "threshold"
        ]
    )
    baseline_selected = control_bot_score >= control_bot_threshold
    quality_selected = quality_bot_score >= quality_bot_threshold
    two_stage_selected, combined_score = _two_stage_selection(
        quality_bot_score, economic_score, int(quality_selected.sum())
    )
    selective_count = max(1, int(round(0.25 * quality_selected.sum())))
    finite_combined = np.flatnonzero(np.isfinite(combined_score))
    selective_indices = finite_combined[
        np.argpartition(combined_score[finite_combined], -selective_count)[
            -selective_count:
        ]
    ]
    selective_two_stage_selected = np.zeros(len(june), dtype=bool)
    selective_two_stage_selected[selective_indices] = True
    june_predictions = pd.DataFrame(
        {
            "token_address": june.token_address,
            "block_time": june.block_time,
            "label": june.label.astype("uint8"),
            "bot_score": control_bot_score,
            "quality_bot_score": quality_bot_score,
            "economic_score": economic_score,
            "combined_score": combined_score,
            "baseline_selected": baseline_selected.astype("uint8"),
            "quality_selected": quality_selected.astype("uint8"),
            "two_stage_selected": two_stage_selected.astype("uint8"),
            "selective_two_stage_selected": selective_two_stage_selected.astype(
                "uint8"
            ),
        }
    )
    june_predictions.to_parquet(
        ARTIFACTS / "tables" / "third_pass_june_strategy_predictions.parquet",
        index=False,
    )
    outcomes = _build_custom_strategy_outcomes(june_predictions)
    stake, _, fee = _strategy_parameters()
    june_backtest: dict[str, object] = {}
    for policy in ("immediate", "offset_118"):
        policy_frame = outcomes[outcomes.policy.eq(policy)]
        policy_result: dict[str, object] = {}
        for cohort, column in (
            ("baseline_replica", "baseline_selected"),
            ("quality_augmented_replica", "quality_selected"),
            ("two_stage", "two_stage_selected"),
            ("selective_two_stage", "selective_two_stage_selected"),
            ("target_equal_stake", "label"),
        ):
            expected = int(june_predictions[column].sum())
            values, _ = backtest_metrics(
                policy_frame[policy_frame[column].eq(1)], expected, stake, fee
            )
            policy_result[cohort] = values
        june_backtest[policy] = policy_result
    cohorts = _disagreement_characterization(june_predictions, outcomes)
    output["june_reporting_only"] = {
        "marginal_accounting": "network-cost-adjusted and gross of proportional Pump swap fees; inclusive network cost is subtracted once",
        "selection": {
            "baseline": _strategy_selection_summary(
                june.assign(
                    target_label=june.label,
                    label=np.zeros(len(june), dtype=np.uint8),
                    creator_fee_usd_7d=0,
                ),
                baseline_selected,
            ),
            "quality_augmented": _strategy_selection_summary(
                june.assign(
                    target_label=june.label,
                    label=np.zeros(len(june), dtype=np.uint8),
                    creator_fee_usd_7d=0,
                ),
                quality_selected,
            ),
            "two_stage": _strategy_selection_summary(
                june.assign(
                    target_label=june.label,
                    label=np.zeros(len(june), dtype=np.uint8),
                    creator_fee_usd_7d=0,
                ),
                two_stage_selected,
            ),
            "selective_two_stage": _strategy_selection_summary(
                june.assign(
                    target_label=june.label,
                    label=np.zeros(len(june), dtype=np.uint8),
                    creator_fee_usd_7d=0,
                ),
                selective_two_stage_selected,
            ),
        },
        "backtest": june_backtest,
        "disagreement_cohorts": cohorts.to_dict("records"),
    }
    ECONOMIC_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output
