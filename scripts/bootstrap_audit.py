#!/usr/bin/env python3
"""Read-only inventory and relationship audit for the local competition data.

Run the inexpensive footer inventory with:

    uv run --no-project --with duckdb==1.5.5 scripts/bootstrap_audit.py

Add ``--relationships`` to reproduce the narrow key-column scans reported in
``docs/DATA.md``. The script never writes to ``data/raw``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def row_dict(cursor: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    names = [column[0] for column in cursor.description]
    return dict(zip(names, cursor.fetchone(), strict=True))


def parquet_inventory(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, Any]:
    metadata = row_dict(
        con.execute(
            """
            SELECT num_rows, num_row_groups, format_version, created_by,
                   file_size_bytes
            FROM parquet_file_metadata(?)
            """,
            [str(path)],
        )
    )
    described = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall()
    schema = [{"name": row[0], "type": row[1]} for row in described]
    temporal = {}
    for column in schema:
        name = column["name"]
        if "time" not in name.lower() and "slot" not in name.lower():
            continue
        if "TIMESTAMP" in column["type"]:
            result = con.execute(
                """
                SELECT min(stats_min_value), max(stats_max_value),
                       sum(stats_null_count)
                FROM parquet_metadata(?)
                WHERE path_in_schema = ?
                """,
                [str(path), name],
            ).fetchone()
        else:
            result = con.execute(
                """
                SELECT min(try_cast(stats_min_value AS BIGINT)),
                       max(try_cast(stats_max_value AS BIGINT)),
                       sum(stats_null_count)
                FROM parquet_metadata(?)
                WHERE path_in_schema = ?
                """,
                [str(path), name],
            ).fetchone()
        temporal[name] = {"min": result[0], "max": result[1], "nulls": result[2]}
    return {**metadata, "schema": schema, "temporal_footer_stats": temporal}


def inventory(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(candidate for candidate in RAW.rglob("*") if candidate.is_file()):
        relative = str(path.relative_to(ROOT))
        item: dict[str, Any] = {"bytes": path.stat().st_size}
        if path.suffix == ".parquet":
            item["format"] = "parquet"
            item.update(parquet_inventory(con, path))
        elif path.name.endswith(".jsonl.gz"):
            item["format"] = "gzip-compressed JSON Lines"
        elif path.suffix == ".jsonl":
            item["format"] = "JSON Lines"
        else:
            item["format"] = "unknown"
        result[relative] = item
    return result


def relationship_audit(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    bought_index = sql_path(RAW / "core" / "bought_deploy_txs_index.parquet")
    negative_index = sql_path(
        RAW / "core" / "not_bought_deploy_txs_index.parquet"
    )
    target_index = sql_path(
        RAW / "target_wallet" / "5brv79e_activity_txs_index.parquet"
    )
    bought_activity = sql_path(
        RAW / "core" / "bought_deployers_activity.parquet"
    )
    negative_activity = sql_path(
        RAW / "core" / "not_bought_deployers_activity.parquet"
    )
    target_activity = sql_path(
        RAW / "target_wallet" / "5brv79e_activity.parquet"
    )
    trades = sql_path(RAW / "june" / "trades" / "pumpfun_trades.parquet")
    candles = sql_path(RAW / "june" / "candles" / "mcap_candles.parquet")
    jito_transactions = sql_path(
        RAW / "june" / "jito" / "jito_bundle_transactions.parquet"
    )
    jito_bundles = sql_path(RAW / "june" / "jito" / "jito_bundles.parquet")
    jito_tippers = sql_path(
        RAW / "june" / "jito" / "jito_bundle_tippers.parquet"
    )

    result: dict[str, Any] = {}
    for label, path in {
        "bought_index": bought_index,
        "not_bought_index": negative_index,
        "target_wallet_index": target_index,
    }.items():
        columns = {
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).fetchall()
        }
        select = [
            "count(*) AS n_rows",
            "count(DISTINCT tx_hash) AS distinct_tx_hash",
            "count(DISTINCT line_number) AS distinct_line_number",
            "min(line_number) AS min_line_number",
            "max(line_number) AS max_line_number",
        ]
        if "token_address" in columns:
            select.extend(
                [
                    "count(DISTINCT token_address) AS distinct_tokens",
                    "count(DISTINCT tx_signer) AS distinct_signers",
                    "count(creator_address) AS nonnull_creator_rows",
                    "count(DISTINCT creator_address) AS distinct_creators",
                ]
            )
        result[label] = row_dict(
            con.execute(
                f"SELECT {', '.join(select)} FROM read_parquet('{path}')"
            )
        )

    result["class_overlap"] = row_dict(
        con.execute(
            f"""
            SELECT (
                       SELECT count(*)
                       FROM read_parquet('{bought_index}') b
                       JOIN read_parquet('{negative_index}') n USING (tx_hash)
                   ) AS tx_hash_overlap,
                   (
                       SELECT count(*) FROM (
                           SELECT DISTINCT token_address
                           FROM read_parquet('{bought_index}')
                           INTERSECT
                           SELECT DISTINCT token_address
                           FROM read_parquet('{negative_index}')
                       )
                   ) AS token_overlap
            """
        )
    )
    result["not_bought_duplicate_mappings"] = row_dict(
        con.execute(
            f"""
            WITH tx_duplicates AS (
                SELECT tx_hash, count(*) AS n
                FROM read_parquet('{negative_index}')
                GROUP BY tx_hash
                HAVING count(*) > 1
            ), line_duplicates AS (
                SELECT line_number, count(*) AS n
                FROM read_parquet('{negative_index}')
                GROUP BY line_number
                HAVING count(*) > 1
            ), full_duplicates AS (
                SELECT *, count(*) AS n
                FROM read_parquet('{negative_index}')
                GROUP BY ALL
                HAVING count(*) > 1
            )
            SELECT (SELECT count(*) FROM tx_duplicates)
                       AS duplicate_tx_hash_groups,
                   (SELECT count(*) FROM line_duplicates)
                       AS duplicate_line_number_groups,
                   (SELECT coalesce(sum(n - 1), 0) FROM tx_duplicates)
                       AS extra_deployment_rows,
                   (SELECT max(n) FROM tx_duplicates)
                       AS max_tokens_per_transaction,
                   (SELECT count(*) FROM full_duplicates)
                       AS duplicate_full_row_groups
            """
        )
    )

    result["deployer_wallet_coverage"] = {}
    for label, index_path, activity_path in [
        ("bought", bought_index, bought_activity),
        ("not_bought", negative_index, negative_activity),
    ]:
        result["deployer_wallet_coverage"][label] = row_dict(
            con.execute(
                f"""
                WITH i AS (
                    SELECT DISTINCT tx_signer AS wallet
                    FROM read_parquet('{index_path}')
                ), a AS (
                    SELECT DISTINCT wallet FROM read_parquet('{activity_path}')
                )
                SELECT (SELECT count(*) FROM i) AS deployment_signers,
                       (SELECT count(*) FROM a) AS activity_wallets,
                       (SELECT count(*) FROM (
                           SELECT * FROM i EXCEPT SELECT * FROM a
                       )) AS signers_without_activity,
                       (SELECT count(*) FROM (
                           SELECT * FROM a EXCEPT SELECT * FROM i
                       )) AS activity_wallets_without_deployment
                """
            )
        )

    result["target_wallet"] = row_dict(
        con.execute(
            f"""
            SELECT count(*) AS activity_rows,
                   count(DISTINCT a.tx_hash) AS activity_tx_hashes,
                   count(DISTINCT a.token_address) AS activity_tokens,
                   count(DISTINCT i.tx_hash) AS indexed_raw_transactions,
                   count(DISTINCT a.tx_hash) FILTER (
                       WHERE i.tx_hash IS NOT NULL
                   ) AS activity_txs_with_raw_transaction
            FROM read_parquet('{target_activity}') a
            LEFT JOIN read_parquet('{target_index}') i USING (tx_hash)
            """
        )
    )
    result["target_class_links"] = row_dict(
        con.execute(
            f"""
            WITH a AS (
                SELECT DISTINCT token_address
                FROM read_parquet('{target_activity}')
            )
            SELECT (SELECT count(*) FROM a) AS target_activity_tokens,
                   (SELECT count(*) FROM read_parquet('{bought_index}') b
                    JOIN a USING (token_address)) AS bought_tokens,
                   (SELECT count(*) FROM read_parquet('{negative_index}') n
                    JOIN a USING (token_address)) AS not_bought_tokens,
                   (SELECT count(*) FROM read_parquet('{target_index}') t
                    JOIN read_parquet('{bought_index}') b USING (tx_hash))
                       AS bought_deploy_tx_overlaps
            """
        )
    )
    result["bought_activity"] = row_dict(
        con.execute(
            f"""
            SELECT count(*) AS activity_rows,
                   count(DISTINCT tx_hash) AS transaction_hashes,
                   count(DISTINCT token_address) AS tokens
            FROM read_parquet('{bought_activity}')
            """
        )
    )

    result["june_deployments"] = {}
    for label, path in [("bought", bought_index), ("not_bought", negative_index)]:
        result["june_deployments"][label] = row_dict(
            con.execute(
                f"""
                SELECT count(*) AS deployments,
                       count(DISTINCT tx_hash) AS deployment_transactions,
                       count(DISTINCT token_address) AS tokens
                FROM read_parquet('{path}')
                WHERE blockTime BETWEEN 1780272000 AND 1782863999
                """
            )
        )

    deployment_cte = f"""
        SELECT 'bought' AS class_name, token_address, tx_hash, tx_signer,
               blockTime, blockSlot
        FROM read_parquet('{bought_index}')
        WHERE blockTime BETWEEN 1780272000 AND 1782863999
        UNION ALL
        SELECT 'not_bought', token_address, tx_hash, tx_signer,
               blockTime, blockSlot
        FROM read_parquet('{negative_index}')
        WHERE blockTime BETWEEN 1780272000 AND 1782863999
    """
    result["june_trades"] = row_dict(
        con.execute(
            f"""
            WITH t AS (
                SELECT DISTINCT token_address, deploy_tx_hash, deploy_tx_signer,
                       creator_address, deploy_block_time, deploy_block_slot
                FROM read_parquet('{trades}')
            ), d AS ({deployment_cte})
            SELECT count(*) AS trade_deployments,
                   count(*) FILTER (WHERE d.class_name = 'bought') AS bought,
                   count(*) FILTER (WHERE d.class_name = 'not_bought') AS not_bought,
                   count(*) FILTER (WHERE d.class_name IS NULL) AS unmatched,
                   count(*) FILTER (
                       WHERE t.creator_address IS DISTINCT FROM t.deploy_tx_signer
                   ) AS creator_signer_mismatches
            FROM t
            LEFT JOIN d
              ON t.deploy_tx_hash = d.tx_hash
             AND t.token_address = d.token_address
            """
        )
    )

    result["june_candles"] = row_dict(
        con.execute(
            f"""
            WITH m AS (
                SELECT token_address, min(deploy_time_s) AS deploy_time,
                       min(candle_time_s) AS first_candle, count(*) AS candles
                FROM read_parquet('{candles}')
                GROUP BY token_address
            ), d AS ({deployment_cte})
            SELECT count(*) AS candle_tokens, sum(candles) AS candle_rows,
                   count(*) FILTER (WHERE d.class_name = 'bought') AS bought,
                   count(*) FILTER (WHERE d.class_name = 'not_bought') AS not_bought,
                   count(*) FILTER (WHERE d.class_name IS NULL) AS unmatched,
                   count(*) FILTER (
                       WHERE first_candle < deploy_time
                   ) AS tokens_with_predeploy_candle
            FROM m LEFT JOIN d USING (token_address)
            """
        )
    )
    result["candle_trade_coverage"] = row_dict(
        con.execute(
            f"""
            WITH m AS (
                SELECT DISTINCT token_address FROM read_parquet('{candles}')
            ), t AS (
                SELECT DISTINCT token_address FROM read_parquet('{trades}')
            )
            SELECT (SELECT count(*) FROM m LEFT JOIN t USING (token_address)
                    WHERE t.token_address IS NULL) AS candles_without_trades,
                   (SELECT count(*) FROM t LEFT JOIN m USING (token_address)
                    WHERE m.token_address IS NULL) AS trades_without_candles
            """
        )
    )
    result["predeploy_candle_rows"] = row_dict(
        con.execute(
            f"""
            SELECT count(*) AS predeploy_rows,
                   min(candle_time_s - deploy_time_s) AS min_seconds,
                   max(candle_time_s - deploy_time_s) AS max_seconds
            FROM read_parquet('{candles}')
            WHERE candle_time_s < deploy_time_s
            """
        )
    )

    result["jito_deployment_links"] = row_dict(
        con.execute(
            f"""
            WITH d AS ({deployment_cte}), m AS (
                SELECT d.*, j.bundle_id, j.slot AS jito_slot
                FROM d
                JOIN read_parquet('{jito_transactions}') j
                  ON d.tx_hash = j.tx_signature
            )
            SELECT count(*) AS association_rows,
                   count(DISTINCT tx_hash) AS deployment_transactions,
                   count(DISTINCT token_address) AS tokens,
                   count(DISTINCT token_address) FILTER (
                       WHERE class_name = 'bought'
                   ) AS bought_tokens,
                   count(DISTINCT token_address) FILTER (
                       WHERE class_name = 'not_bought'
                   ) AS not_bought_tokens,
                   count(*) FILTER (WHERE blockSlot <> jito_slot) AS slot_mismatches
            FROM m
            """
        )
    )
    result["matched_jito_bundle_coverage"] = row_dict(
        con.execute(
            f"""
            WITH d AS ({deployment_cte}), m AS (
                SELECT DISTINCT j.bundle_id, j.slot
                FROM d
                JOIN read_parquet('{jito_transactions}') j
                  ON d.tx_hash = j.tx_signature
            ), b AS (
                SELECT bundle_id, slot FROM read_parquet('{jito_bundles}')
            ), t AS (
                SELECT bundle_id, slot FROM read_parquet('{jito_tippers}')
            )
            SELECT count(*) AS matched_bundle_keys,
                   count(*) FILTER (WHERE b.bundle_id IS NULL)
                       AS missing_bundle_details,
                   count(*) FILTER (WHERE t.bundle_id IS NULL)
                       AS missing_tipper
            FROM m
            LEFT JOIN b USING (bundle_id, slot)
            LEFT JOIN t USING (bundle_id, slot)
            """
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relationships",
        action="store_true",
        help="also run narrow scans that validate joins and duplicate behavior",
    )
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '3GB'")
    con.execute("SET temp_directory = '/tmp/solana-bootstrap-duckdb'")
    output: dict[str, Any] = {"inventory": inventory(con)}
    if args.relationships:
        output["relationships"] = relationship_audit(con)
    print(json.dumps(output, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
