from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .config import (
    BOUGHT_ACTIVITY,
    BOUGHT_INDEX,
    INTERIM,
    NOT_BOUGHT_ACTIVITY,
    NOT_BOUGHT_INDEX,
    PROCESSED,
    SUBMISSION,
    ensure_output_dirs,
)


MESSAGE_DIR = INTERIM / "message_features"
BASE_FEATURES = INTERIM / "deployment_base_features.parquet"
ACTIVITY_WALLETS = INTERIM / "not_bought_activity_wallets.parquet"
ACTIVITY_STATE = INTERIM / "deployer_activity_state.parquet"
FEATURE_STORE = PROCESSED / "deployment_features.parquet"
FEATURE_MANIFEST = PROCESSED / "deployment_features.manifest.json"


def _sql(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='36GB'")
    con.execute("SET threads=20")
    con.execute("SET preserve_insertion_order=false")
    temp = INTERIM / "duckdb_temp"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{_sql(temp)}'")
    return con


def build_base(force: bool = False) -> Path:
    if BASE_FEATURES.exists() and not force:
        return BASE_FEATURES
    bought_messages = MESSAGE_DIR / "bought.parquet"
    negative_messages = MESSAGE_DIR / "not_bought.parquet"
    missing = [p for p in (bought_messages, negative_messages) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run message feature extraction first; missing {missing}")
    con = connection()
    start = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH deployments AS (
            SELECT token_address, tx_hash, tx_signer, creator_address,
                   blockTime AS block_time, blockSlot AS block_slot, 1::UTINYINT AS label
            FROM read_parquet('{_sql(BOUGHT_INDEX)}')
            UNION ALL
            SELECT token_address, tx_hash, tx_signer, creator_address,
                   blockTime, blockSlot, 0::UTINYINT
            FROM read_parquet('{_sql(NOT_BOUGHT_INDEX)}')
          ), messages AS (
            SELECT * FROM read_parquet('{_sql(bought_messages)}')
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{_sql(negative_messages)}')
          ), deploy_time_groups AS (
            SELECT tx_signer, block_time, count(*) AS deployments_at_second
            FROM deployments GROUP BY 1,2
          ), deploy_history AS (
            SELECT *,
              coalesce(sum(deployments_at_second) OVER (
                PARTITION BY tx_signer ORDER BY block_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS prior_deploy_count,
              lag(block_time) OVER (PARTITION BY tx_signer ORDER BY block_time) AS prior_deploy_time,
              coalesce(sum(deployments_at_second) OVER (
                PARTITION BY tx_signer ORDER BY block_time
                RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING), 0) AS prior_deploy_count_1h,
              coalesce(sum(deployments_at_second) OVER (
                PARTITION BY tx_signer ORDER BY block_time
                RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING), 0) AS prior_deploy_count_1d,
              coalesce(sum(deployments_at_second) OVER (
                PARTITION BY tx_signer ORDER BY block_time
                RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING), 0) AS prior_deploy_count_7d,
              coalesce(sum(deployments_at_second) OVER (
                PARTITION BY tx_signer ORDER BY block_time
                RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING), 0) AS prior_deploy_count_30d
            FROM deploy_time_groups
          ), joined AS (
            SELECT d.*, m.* EXCLUDE (tx_hash, token_address, block_time, block_slot),
                   (m.tx_hash IS NULL)::UTINYINT AS message_missing,
                   h.deployments_at_second, h.prior_deploy_count,
                   h.prior_deploy_count_1h, h.prior_deploy_count_1d,
                   h.prior_deploy_count_7d, h.prior_deploy_count_30d,
                   d.block_time - h.prior_deploy_time AS seconds_since_prior_deploy,
                   extract(hour FROM to_timestamp(d.block_time))::SMALLINT AS deploy_hour_utc,
                   extract(dow FROM to_timestamp(d.block_time))::SMALLINT AS deploy_day_of_week_utc,
                   (d.block_time - 1767225600) / 86400.0 AS days_since_2026_start,
                   least(coalesce(m.dev_buy_lamports, 0) / 1000000000.0, 1000.0) AS dev_buy_sol,
                   (coalesce(m.dev_buy_lamports, 0) > 1000000000000)::UTINYINT AS dev_buy_over_1000_sol,
                   coalesce(m.system_transfer_lamports, 0) / 1000000000.0 AS system_transfer_sol,
                   coalesce(m.max_system_transfer_lamports, 0) / 1000000000.0 AS max_system_transfer_sol
            FROM deployments d
            LEFT JOIN messages m USING (tx_hash, token_address)
            JOIN deploy_history h
              ON d.tx_signer=h.tx_signer AND d.block_time=h.block_time
          ), name_groups AS (
            SELECT name_normalized, block_time, count(*) AS names_at_second
            FROM joined WHERE name_normalized IS NOT NULL
            GROUP BY 1,2
          ), name_history AS (
            SELECT *,
              coalesce(sum(names_at_second) OVER (
                PARTITION BY name_normalized ORDER BY block_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS prior_name_count,
              lag(block_time) OVER (PARTITION BY name_normalized ORDER BY block_time) AS prior_name_time
            FROM name_groups
          ), symbol_groups AS (
            SELECT symbol_normalized, block_time, count(*) AS symbols_at_second
            FROM joined WHERE symbol_normalized IS NOT NULL
            GROUP BY 1,2
          ), symbol_history AS (
            SELECT *,
              coalesce(sum(symbols_at_second) OVER (
                PARTITION BY symbol_normalized ORDER BY block_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS prior_symbol_count,
              lag(block_time) OVER (PARTITION BY symbol_normalized ORDER BY block_time) AS prior_symbol_time
            FROM symbol_groups
          ), signer_name_groups AS (
            SELECT tx_signer, name_normalized, block_time, count(*) AS names_at_second
            FROM joined WHERE name_normalized IS NOT NULL GROUP BY 1,2,3
          ), signer_name_history AS (
            SELECT *, coalesce(sum(names_at_second) OVER (
              PARTITION BY tx_signer, name_normalized ORDER BY block_time
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS signer_prior_name_count
            FROM signer_name_groups
          ), signer_symbol_groups AS (
            SELECT tx_signer, symbol_normalized, block_time, count(*) AS symbols_at_second
            FROM joined WHERE symbol_normalized IS NOT NULL GROUP BY 1,2,3
          ), signer_symbol_history AS (
            SELECT *, coalesce(sum(symbols_at_second) OVER (
              PARTITION BY tx_signer, symbol_normalized ORDER BY block_time
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS signer_prior_symbol_count
            FROM signer_symbol_groups
          )
          SELECT j.*,
                 coalesce(n.prior_name_count, 0) AS prior_name_count,
                 j.block_time - n.prior_name_time AS seconds_since_prior_name,
                 coalesce(s.prior_symbol_count, 0) AS prior_symbol_count,
                 j.block_time - s.prior_symbol_time AS seconds_since_prior_symbol,
                 coalesce(sn.signer_prior_name_count, 0) AS signer_prior_name_count,
                 coalesce(ss.signer_prior_symbol_count, 0) AS signer_prior_symbol_count
          FROM joined j
          LEFT JOIN name_history n USING (name_normalized, block_time)
          LEFT JOIN symbol_history s USING (symbol_normalized, block_time)
          LEFT JOIN signer_name_history sn USING (tx_signer, name_normalized, block_time)
          LEFT JOIN signer_symbol_history ss USING (tx_signer, symbol_normalized, block_time)
        ) TO '{_sql(BASE_FEATURES)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    elapsed = time.monotonic() - start
    print(f"built {BASE_FEATURES} in {elapsed:.1f}s")
    return BASE_FEATURES


def build_activity_wallets(force: bool = False) -> Path:
    if ACTIVITY_WALLETS.exists() and not force:
        return ACTIVITY_WALLETS
    con = connection()
    con.execute(
        f"COPY (SELECT DISTINCT wallet FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')) "
        f"TO '{_sql(ACTIVITY_WALLETS)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return ACTIVITY_WALLETS


def build_activity_state(force: bool = False) -> Path:
    if ACTIVITY_STATE.exists() and not force:
        return ACTIVITY_STATE
    build_activity_wallets(force=force)
    con = connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH raw_activity AS (
            SELECT wallet, timestamp, event_type, is_open_or_close, tx_hash,
                   launchpad, quote_token_symbol, quote_amount, cost_usd,
                   gas_native, priority_fee, tip_fee
            FROM read_parquet('{_sql(NOT_BOUGHT_ACTIVITY)}')
            UNION ALL
            SELECT b.wallet, b.timestamp, b.event_type, b.is_open_or_close, b.tx_hash,
                   b.launchpad, b.quote_token_symbol, b.quote_amount, b.cost_usd,
                   b.gas_native, b.priority_fee, b.tip_fee
            FROM read_parquet('{_sql(BOUGHT_ACTIVITY)}') b
            ANTI JOIN read_parquet('{_sql(ACTIVITY_WALLETS)}') n USING (wallet)
          ), per_second AS (
            SELECT wallet, timestamp,
              count(*)::BIGINT AS event_rows,
              count(DISTINCT tx_hash)::BIGINT AS transactions,
              count(*) FILTER (WHERE event_type='buy')::BIGINT AS buys,
              count(*) FILTER (WHERE event_type='sell')::BIGINT AS sells,
              count(*) FILTER (WHERE event_type='launch')::BIGINT AS launches,
              count(*) FILTER (WHERE event_type='burn')::BIGINT AS burns,
              count(*) FILTER (WHERE is_open_or_close=1)::BIGINT AS opens_or_closes,
              count(*) FILTER (WHERE launchpad='pump')::BIGINT AS pump_events,
              coalesce(sum(try_cast(cost_usd AS DOUBLE)), 0) AS cost_usd,
              coalesce(sum(try_cast(quote_amount AS DOUBLE)) FILTER (
                WHERE quote_token_symbol IN ('SOL','WSOL')), 0) AS quote_sol,
              coalesce(sum(try_cast(gas_native AS DOUBLE)), 0) AS gas_native,
              coalesce(sum(try_cast(priority_fee AS DOUBLE)), 0) AS priority_fee,
              coalesce(sum(try_cast(tip_fee AS DOUBLE)), 0) AS tip_fee
            FROM raw_activity
            WHERE timestamp IS NOT NULL
            GROUP BY wallet, timestamp
          )
          SELECT wallet, timestamp,
            min(timestamp) OVER w AS first_observed_activity_time,
            sum(event_rows) OVER w AS hist_event_count,
            sum(transactions) OVER w AS hist_tx_count,
            sum(buys) OVER w AS hist_buy_count,
            sum(sells) OVER w AS hist_sell_count,
            sum(launches) OVER w AS hist_launch_count,
            sum(burns) OVER w AS hist_burn_count,
            sum(opens_or_closes) OVER w AS hist_open_close_count,
            sum(pump_events) OVER w AS hist_pump_event_count,
            sum(cost_usd) OVER w AS hist_cost_usd_sum,
            sum(quote_sol) OVER w AS hist_quote_sol_sum,
            sum(gas_native) OVER w AS hist_gas_native_sum,
            sum(priority_fee) OVER w AS hist_priority_fee_sum,
            sum(tip_fee) OVER w AS hist_tip_fee_sum
          FROM per_second
          WINDOW w AS (
            PARTITION BY wallet ORDER BY timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          )
          ORDER BY wallet, timestamp
        ) TO '{_sql(ACTIVITY_STATE)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )
    print(f"built {ACTIVITY_STATE} in {time.monotonic() - started:.1f}s")
    return ACTIVITY_STATE


def build_feature_store(force: bool = False, with_activity: bool = True) -> Path:
    build_base(force=force)
    if not with_activity:
        return BASE_FEATURES
    build_activity_state(force=force)
    if FEATURE_STORE.exists() and not force:
        return FEATURE_STORE
    con = connection()
    started = time.monotonic()
    cumulative = [
        "hist_event_count",
        "hist_tx_count",
        "hist_buy_count",
        "hist_sell_count",
        "hist_launch_count",
        "hist_burn_count",
        "hist_open_close_count",
        "hist_pump_event_count",
        "hist_cost_usd_sum",
        "hist_quote_sol_sum",
        "hist_gas_native_sum",
        "hist_priority_fee_sum",
        "hist_tip_fee_sum",
    ]
    current = ",\n".join(f"coalesce(a.{c}, 0) AS {c}" for c in cumulative)
    rolling = ",\n".join(
        f"coalesce(a.{c},0)-coalesce(a1.{c},0) AS {c}_1d, "
        f"coalesce(a.{c},0)-coalesce(a7.{c},0) AS {c}_7d, "
        f"coalesce(a.{c},0)-coalesce(a30.{c},0) AS {c}_30d"
        for c in cumulative[:8]
    )
    con.execute(
        f"""
        COPY (
          SELECT d.*,
                 (a.wallet IS NULL)::UTINYINT AS history_missing,
                 d.block_time - a.first_observed_activity_time AS observed_wallet_age_seconds,
                 d.block_time - a.timestamp AS seconds_since_activity,
                 {current},
                 {rolling}
          FROM read_parquet('{_sql(BASE_FEATURES)}') d
          ASOF LEFT JOIN read_parquet('{_sql(ACTIVITY_STATE)}') a
            ON d.tx_signer=a.wallet AND d.block_time>a.timestamp
          ASOF LEFT JOIN read_parquet('{_sql(ACTIVITY_STATE)}') a1
            ON d.tx_signer=a1.wallet AND d.block_time-86400>a1.timestamp
          ASOF LEFT JOIN read_parquet('{_sql(ACTIVITY_STATE)}') a7
            ON d.tx_signer=a7.wallet AND d.block_time-604800>a7.timestamp
          ASOF LEFT JOIN read_parquet('{_sql(ACTIVITY_STATE)}') a30
            ON d.tx_signer=a30.wallet AND d.block_time-2592000>a30.timestamp
        ) TO '{_sql(FEATURE_STORE)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    elapsed = time.monotonic() - started
    parquet = pq.ParquetFile(FEATURE_STORE)
    manifest = {
        "output": str(FEATURE_STORE),
        "row_count": parquet.metadata.num_rows,
        "schema": str(parquet.schema_arrow),
        "sources": [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in (BOUGHT_INDEX, NOT_BOUGHT_INDEX, BOUGHT_ACTIVITY, NOT_BOUGHT_ACTIVITY)
        ],
        "decision_clock": "deployment block_time; activity.timestamp < block_time",
        "same_second_policy": "excluded for activity and deployment-history windows",
        "message_policy": "signed transaction.message only; meta excluded",
        "elapsed_seconds_final_join": elapsed,
        "command": "python -m solana_sniper_reverse_engineering.feature_store",
    }
    FEATURE_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_feature_dictionary()
    write_source_roles()
    print(json.dumps({k: v for k, v in manifest.items() if k != "schema"}, indent=2))
    return FEATURE_STORE


def write_source_roles() -> None:
    roles = {
        "A_entry_features": {
            "sources": ["deployment indexes", "signed transaction.message", "strictly earlier deployer activity"],
            "boundary": "activity.timestamp < deployment.blockTime",
        },
        "B_labels_evaluation": ["bought/not-bought membership", "target-wallet activity"],
        "C_backtest_only": ["June pump.fun trades", "June market-cap candles"],
        "D_behavior_only": ["full target history", "post-deployment deployer events", "landed Jito context"],
        "E_excluded_ambiguous": ["raw transaction meta", "same-second activity", "creator_address semantics", "landed bundle facts"],
    }
    (SUBMISSION / "tables" / "source_roles.json").write_text(
        json.dumps(roles, indent=2, sort_keys=True) + "\n"
    )


def write_feature_dictionary() -> None:
    rows = [
        ("deployment cadence", "prior_deploy_count*; seconds_since_prior_deploy", "both core indexes", "strict earlier deployment second", "zero/null", "known before current deployment", "deployer repetition/cadence"),
        ("timing", "deploy_hour_utc; deploy_day_of_week_utc; days_since_2026_start", "deployment index", "current deployment clock", "none", "at t_decision", "time/regime policy"),
        ("metadata text", "name/symbol length, case, digits, Unicode, equality; URI provider", "signed create instruction", "current signed message", "missing flag", "message exists by t_decision", "metadata screening"),
        ("metadata reuse", "prior_*name/symbol*", "earlier signed create instructions", "grouped windows ending one second before", "zero/null recency", "strictly earlier deployments", "global/deployer reuse"),
        ("dev buy", "has_dev_buy; dev_buy_sol; instruction kind/arguments", "signed Pump instruction", "current signed message", "zero/none", "message exists by t_decision", "developer commitment"),
        ("message structure", "instruction/account/signer/writable/lookup counts", "signed transaction.message", "current signed message", "missing flag", "message exists by t_decision", "transaction construction"),
        ("compute budget", "limit; price; instruction count", "signed ComputeBudget instructions", "current signed message", "null/zero", "message exists by t_decision", "priority intent"),
        ("top-level transfers", "count/sum/max SOL", "signed parsed System instructions", "current signed message", "zero", "message exists by t_decision", "tips/funding-like transfers; destination semantics unresolved"),
        ("historical activity", "cumulative prior event/tx/type/volume/fee counts", "deployer activity", "ASOF activity.timestamp < block_time", "zero + history_missing", "only strictly earlier rows", "wallet experience/activity"),
        ("rolling activity", "1d/7d/30d differences of cumulative counts", "deployer activity", "[t-window,t) via strict ASOF states", "zero", "only strictly earlier rows", "recent activity intensity"),
        ("observed age/recency", "observed_wallet_age_seconds; seconds_since_activity", "deployer activity", "strict prior state", "null", "only prior rows, but snapshot can be left-censored", "wallet tenure/recency"),
    ]
    header = "family,features,source,temporal_construction,missing_policy,legality,interpretation\n"
    lines = [header]
    import csv
    import io

    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["family", "features", "source", "temporal_construction", "missing_policy", "legality", "interpretation"])
    writer.writerows(rows)
    (SUBMISSION / "tables" / "feature_dictionary.csv").write_text(stream.getvalue())


def audit(path: Path = FEATURE_STORE) -> dict[str, object]:
    con = connection()
    result = con.execute(
        f"""
        SELECT count(*) AS rows, count(DISTINCT token_address) AS tokens,
               count(*)-count(DISTINCT (tx_hash, token_address)) AS duplicate_keys,
               sum(label) AS positives,
               count(*) FILTER (WHERE history_missing=1) AS missing_history,
               count(*) FILTER (WHERE message_missing=1) AS missing_message,
               count(*) FILTER (WHERE seconds_since_activity < 1) AS invalid_activity_recency,
               count(*) FILTER (WHERE seconds_since_prior_deploy < 1) AS invalid_deploy_recency
        FROM read_parquet('{_sql(path)}')
        """
    )
    names = [column[0] for column in result.description]
    values = result.fetchone()
    output = dict(zip(names, values, strict=True))
    print(json.dumps(output, indent=2))
    return output


def run(force: bool = False, without_activity: bool = False) -> Path:
    ensure_output_dirs()
    path = build_feature_store(force=force, with_activity=not without_activity)
    if not without_activity:
        checks = audit(path)
        if checks["rows"] != 5_076_421 or checks["positives"] != 15_927:
            raise RuntimeError(f"feature-store cardinality failure: {checks}")
        if checks["duplicate_keys"] or checks["invalid_activity_recency"] or checks["invalid_deploy_recency"]:
            raise RuntimeError(f"feature-store temporal/key failure: {checks}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the leakage-safe deployment feature store")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--without-activity", action="store_true")
    args = parser.parse_args()
    run(force=args.force, without_activity=args.without_activity)


if __name__ == "__main__":
    main()
