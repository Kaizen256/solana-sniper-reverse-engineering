from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    ARTIFACTS,
    BOUGHT_INDEX,
    INTERIM,
    JITO_TRANSACTIONS,
    SUBMISSION,
    TARGET_ACTIVITY,
    TARGET_TXS,
    ensure_output_dirs,
)
from .fee_ledger import PUMP_TARGET_ROUTE_TOTAL_RATE, build_transaction_fee_ledger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TARGET_POSITIONS = INTERIM / "target_tx_positions.parquet"
TOKEN_METRICS = ARTIFACTS / "tables" / "target_token_metrics.parquet"
LATENCIES = ARTIFACTS / "tables" / "target_entry_latencies.parquet"


def _write_target_positions(force: bool = False) -> Path:
    if TARGET_POSITIONS.exists() and not force:
        return TARGET_POSITIONS
    schema = pa.schema(
        [
            ("tx_hash", pa.string()),
            ("block_time", pa.int64()),
            ("block_slot", pa.int64()),
            ("transaction_index", pa.int32()),
        ]
    )
    writer = pq.ParquetWriter(TARGET_POSITIONS, schema, compression="zstd")
    batch: list[dict[str, object]] = []
    try:
        with gzip.open(TARGET_TXS, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                batch.append(
                    {
                        "tx_hash": record["transaction"]["signatures"][0],
                        "block_time": record["blockTime"],
                        "block_slot": record["slot"],
                        "transaction_index": record["transactionIndex"],
                    }
                )
                if len(batch) == 20_000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    finally:
        writer.close()
    return TARGET_POSITIONS


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _quantiles(series: pd.Series) -> dict[str, float | int | None]:
    x = series.dropna().astype(float)
    if x.empty:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)),
        "iqr": float(x.quantile(0.75) - x.quantile(0.25)),
        "p10": float(x.quantile(0.10)),
        "p90": float(x.quantile(0.90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _save_figure(fig: plt.Figure, name: str) -> None:
    artifact = ARTIFACTS / "figures" / name
    public = SUBMISSION / "figures" / name
    fig.savefig(artifact, dpi=180, bbox_inches="tight", facecolor="white")
    shutil.copy2(artifact, public)
    plt.close(fig)


def _target_activity() -> pd.DataFrame:
    columns = [
        "timestamp",
        "event_type",
        "is_open_or_close",
        "tx_hash",
        "token_address",
        "token_symbol",
        "quote_amount",
        "quote_token_symbol",
        "cost_usd",
        "gas_usd",
        "gas_native",
        "dex_usd",
        "dex_native",
        "priority_fee",
        "tip_fee",
        "token_amount",
        "launchpad",
        "launchpad_platform",
    ]
    con = duckdb.connect()
    quoted = ", ".join(columns)
    frame = con.execute(
        f"SELECT {quoted} FROM read_parquet(?) ORDER BY timestamp, tx_hash",
        [str(TARGET_ACTIVITY)],
    ).fetch_df()
    _numeric(
        frame,
        [
            "quote_amount",
            "cost_usd",
            "gas_usd",
            "gas_native",
            "dex_usd",
            "dex_native",
            "priority_fee",
            "tip_fee",
            "token_amount",
        ],
    )
    return frame


def _build_token_metrics(activity: pd.DataFrame) -> pd.DataFrame:
    trade = activity[activity.event_type.isin(["buy", "sell"])].copy()
    # The canonical ledger is transaction-level, so inclusive gas and any
    # separately charged Pump cost can only be allocated once.
    tx_fees = build_transaction_fee_ledger(activity)
    fee_by_token = (
        tx_fees.groupby("token_address", as_index=False)
        .agg(
            network_cost_usd=("network_cost_usd", "sum"),
            network_cost_native=("network_cost_native", "sum"),
            pump_separate_cost_usd=("pump_separate_cost_usd", "sum"),
            pump_separate_cost_native=("pump_separate_cost_native", "sum"),
            pump_reported_component_usd=("pump_reported_component_usd", "sum"),
            pump_raw_observed_fallback_usd=("pump_raw_observed_fallback_usd", "sum"),
            pump_reported_residual_ambiguous_usd=("pump_reported_residual_ambiguous_usd", "sum"),
            routed_dex_contained_in_quote_usd=("routed_dex_contained_in_quote_usd", "sum"),
            unidentified_dex_ambiguous_usd=("unidentified_dex_ambiguous_usd", "sum"),
            priority_fee_included_native=("priority_fee_included_native", "sum"),
            tip_fee_included_native=("tip_fee_included_native", "sum"),
            total_defensible_cost_usd=("total_defensible_cost_usd", "sum"),
            total_defensible_cost_native=("total_defensible_cost_native", "sum"),
        )
    )
    grouped = trade.groupby("token_address", as_index=False).agg(
        symbol=("token_symbol", "first"),
        first_trade_time=("timestamp", "min"),
        first_buy_time=("timestamp", lambda x: x[trade.loc[x.index, "event_type"].eq("buy")].min()),
        last_trade_time=("timestamp", "max"),
        gross_buy_usd=("cost_usd", lambda x: x[trade.loc[x.index, "event_type"].eq("buy")].sum()),
        gross_sell_usd=("cost_usd", lambda x: x[trade.loc[x.index, "event_type"].eq("sell")].sum()),
        buy_transactions=("event_type", lambda x: int(x.eq("buy").sum())),
        sell_transactions=("event_type", lambda x: int(x.eq("sell").sum())),
        partial_sell_transactions=(
            "is_open_or_close",
            lambda x: int(
                (
                    trade.loc[x.index, "event_type"].eq("sell")
                    & pd.Series(x, index=x.index).eq(0)
                ).sum()
            ),
        ),
        close_sell_transactions=(
            "is_open_or_close",
            lambda x: int(
                (
                    trade.loc[x.index, "event_type"].eq("sell")
                    & pd.Series(x, index=x.index).eq(1)
                ).sum()
            ),
        ),
    )
    closes = (
        trade[(trade.event_type == "sell") & (trade.is_open_or_close == 1)]
        .groupby("token_address", as_index=False)
        .agg(final_close_time=("timestamp", "max"))
    )
    burns = (
        activity[activity.event_type == "burn"]
        .groupby("token_address", as_index=False)
        .agg(burn_transactions=("tx_hash", "nunique"), burned_tokens=("token_amount", "sum"))
    )
    result = grouped.merge(closes, on="token_address", how="left")
    result = result.merge(fee_by_token, on="token_address", how="left")
    result = result.merge(burns, on="token_address", how="left")
    fee_columns = [column for column in fee_by_token.columns if column != "token_address"]
    result[[*fee_columns, "burn_transactions", "burned_tokens"]] = result[
        [*fee_columns, "burn_transactions", "burned_tokens"]
    ].fillna(0)
    # Compatibility aliases now mean the full defensible ledger, not gas alone.
    result["fees_usd"] = result.total_defensible_cost_usd
    result["fees_native"] = result.total_defensible_cost_native
    result["hold_seconds"] = result.final_close_time - result.first_buy_time
    result["gross_pnl_usd"] = result.gross_sell_usd - result.gross_buy_usd
    result["net_pnl_usd"] = result.gross_pnl_usd - result.total_defensible_cost_usd
    result["gross_roi"] = result.gross_pnl_usd / result.gross_buy_usd.replace(0, np.nan)
    result["net_roi"] = result.net_pnl_usd / (
        result.gross_buy_usd + result.total_defensible_cost_usd
    ).replace(0, np.nan)
    result.to_parquet(TOKEN_METRICS, index=False)
    return result


def _build_latencies(activity: pd.DataFrame, positions: Path) -> pd.DataFrame:
    buys = (
        activity[activity.event_type == "buy"]
        .sort_values(["timestamp", "tx_hash"])
        .drop_duplicates("token_address", keep="first")
        [["token_address", "tx_hash", "timestamp", "quote_amount", "cost_usd", "quote_token_symbol"]]
        .rename(
            columns={
                "tx_hash": "buy_tx_hash",
                "timestamp": "buy_activity_time",
                "quote_amount": "entry_quote_amount",
                "cost_usd": "entry_cost_usd",
            }
        )
    )
    con = duckdb.connect()
    deployments = con.execute(
        "SELECT token_address, tx_hash AS deploy_tx_hash, blockTime AS deploy_time, "
        "blockSlot AS deploy_slot, tx_signer FROM read_parquet(?)",
        [str(BOUGHT_INDEX)],
    ).fetch_df()
    tx_positions = pd.read_parquet(positions).rename(
        columns={
            "tx_hash": "buy_tx_hash",
            "block_time": "buy_time",
            "block_slot": "buy_slot",
            "transaction_index": "buy_transaction_index",
        }
    )
    result = deployments.merge(buys, on="token_address", how="left", validate="one_to_one")
    result = result.merge(tx_positions, on="buy_tx_hash", how="left", validate="many_to_one")
    # Deployment transactionIndex is message-time evidence in the raw transaction,
    # not stored in the index. Read the small bought file directly.
    dep_pos: list[tuple[str, int]] = []
    with gzip.open(
        Path(BOUGHT_INDEX).with_name("bought_deploy_txs.jsonl.gz"), "rt", encoding="utf-8"
    ) as stream:
        for line in stream:
            record = json.loads(line)
            dep_pos.append(
                (record["transaction"]["signatures"][0], record["transactionIndex"])
            )
    dep_frame = pd.DataFrame(dep_pos, columns=["deploy_tx_hash", "deploy_transaction_index"])
    result = result.merge(dep_frame, on="deploy_tx_hash", how="left", validate="many_to_one")
    result["latency_seconds"] = result.buy_time - result.deploy_time
    result["latency_slots"] = result.buy_slot - result.deploy_slot
    result["same_slot_tx_delta"] = np.where(
        result.latency_slots.eq(0),
        result.buy_transaction_index - result.deploy_transaction_index,
        np.nan,
    )
    result.to_parquet(LATENCIES, index=False)
    return result


def _jito_positioning(latencies: pd.DataFrame) -> pd.DataFrame:
    june = latencies[
        (latencies.deploy_time >= 1_780_272_000)
        & latencies.buy_tx_hash.notna()
    ][["token_address", "deploy_slot", "deploy_tx_hash", "buy_tx_hash"]].copy()
    if june.empty:
        return pd.DataFrame()
    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    con.execute("SET temp_directory='/tmp/solana-behavior-jito'")
    con.register("june_pairs", june)
    result = con.execute(
        f"""
        WITH wanted AS (
            SELECT token_address, deploy_slot AS slot, deploy_tx_hash AS signature,
                   'deploy' AS kind FROM june_pairs
            UNION ALL
            SELECT token_address, deploy_slot, buy_tx_hash, 'buy' FROM june_pairs
        ), matched AS (
            SELECT w.*, j.bundle_id, j.tx_signature_index
            FROM read_parquet('{JITO_TRANSACTIONS.as_posix()}') j
            JOIN wanted w
              ON j.tx_signature = w.signature AND j.slot = w.slot
        ), paired AS (
            SELECT token_address, bundle_id, slot,
                   min(tx_signature_index) FILTER (WHERE kind='deploy') deploy_bundle_index,
                   min(tx_signature_index) FILTER (WHERE kind='buy') buy_bundle_index,
                   count(DISTINCT kind) kinds
            FROM matched GROUP BY 1,2,3
        )
        SELECT *, buy_bundle_index - deploy_bundle_index AS bundle_index_delta
        FROM paired WHERE kinds=2
        """
    ).fetch_df()
    result.to_parquet(ARTIFACTS / "tables" / "june_same_bundle_entries.parquet", index=False)
    return result


def _figures(
    core: pd.DataFrame,
    token_metrics: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    blue, orange = "#176B87", "#E07A2D"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.log10(core.entry_cost_usd.clip(lower=1e-6)), bins=55, color=blue)
    axes[0].set(xlabel="log10 entry value (USD)", ylabel="Tokens", title="Entry-size distribution")
    axes[1].boxplot(core.entry_cost_usd.dropna(), vert=False, showfliers=False)
    axes[1].set(xlabel="Entry value (USD)", title="Central entry-size dispersion")
    _save_figure(fig, "01_entry_size.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(core.latency_slots.clip(-2, 15).dropna(), bins=np.arange(-2.5, 16.5), color=blue)
    axes[0].set(xlabel="Deployment → first buy (slots; clipped at 15)", ylabel="Tokens", title="Entry latency by slot")
    axes[1].hist(core.latency_seconds.clip(-2, 20).dropna(), bins=np.arange(-2.5, 21.5), color=orange)
    axes[1].set(xlabel="Deployment → first buy (seconds; clipped at 20)", title="Entry latency by block time")
    _save_figure(fig, "02_entry_latency.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    hold = token_metrics.hold_seconds.clip(lower=0).dropna()
    axes[0].hist(np.log10(hold + 1), bins=55, color=blue)
    axes[0].set(xlabel="log10(hold seconds + 1)", ylabel="Tokens", title="Holding-time distribution")
    exit_counts = token_metrics.sell_transactions.clip(upper=12).value_counts().sort_index()
    axes[1].bar(exit_counts.index.astype(str), exit_counts.values, color=orange)
    axes[1].set(xlabel="Sell transactions (12 = 12+)", ylabel="Tokens", title="Exit-leg structure")
    _save_figure(fig, "03_holds_and_exits.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pnl = token_metrics.net_pnl_usd.dropna()
    lo, hi = pnl.quantile([0.01, 0.99])
    axes[0].hist(pnl.clip(lo, hi), bins=60, color=blue)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Fully fee-adjusted cash-flow P&L USD (1–99% winsorized)", ylabel="Tokens", title="Per-token P&L")
    axes[1].plot(
        monthly.month,
        monthly.fully_fee_adjusted_pnl_usd.cumsum(),
        marker="o",
        color=orange,
    )
    axes[1].set(xlabel="Cumulative month", ylabel="Fully fee-adjusted P&L (USD)", title="Temporal P&L")
    axes[1].tick_params(axis="x", rotation=30)
    _save_figure(fig, "04_pnl_and_time.png")


def run(force: bool = False) -> dict[str, object]:
    ensure_output_dirs()
    positions = _write_target_positions(force=force)
    activity = _target_activity()
    token_metrics = _build_token_metrics(activity)
    bought_positions = token_metrics[token_metrics.buy_transactions > 0].copy()
    latencies = _build_latencies(activity, positions)
    core_metrics = latencies.merge(token_metrics, on="token_address", how="left", validate="one_to_one")
    core_metrics.to_parquet(ARTIFACTS / "tables" / "core_bought_behavior.parquet", index=False)
    same_bundle = _jito_positioning(latencies)

    first_buys = (
        activity[activity.event_type == "buy"]
        .sort_values(["timestamp", "tx_hash"])
        .drop_duplicates("token_address")
    )
    monthly_activity = activity.copy()
    monthly_activity["month"] = pd.to_datetime(
        monthly_activity.timestamp, unit="s", utc=True
    ).dt.strftime("%Y-%m")
    monthly = (
        monthly_activity.groupby("month", as_index=False)
        .agg(
            activity_rows=("tx_hash", "size"),
            transactions=("tx_hash", "nunique"),
            traded_tokens=("token_address", "nunique"),
            buys=("event_type", lambda x: int(x.eq("buy").sum())),
            sells=("event_type", lambda x: int(x.eq("sell").sum())),
        )
    )
    token_month = bought_positions.copy()
    token_month["month"] = pd.to_datetime(
        token_month.first_buy_time, unit="s", utc=True
    ).dt.strftime("%Y-%m")
    pnl_month = token_month.groupby("month", as_index=False).agg(
        closed_tokens=("token_address", "size"),
        fully_fee_adjusted_pnl_usd=("net_pnl_usd", "sum"),
        fully_fee_adjusted_hit_rate=("net_pnl_usd", lambda x: float((x > 0).mean())),
    )
    monthly = monthly.merge(pnl_month, on="month", how="outer").sort_values("month")
    monthly.to_csv(SUBMISSION / "tables" / "behavior_monthly.csv", index=False)

    winning = bought_positions[bought_positions.net_pnl_usd > 0].net_pnl_usd
    losing = bought_positions[bought_positions.net_pnl_usd <= 0].net_pnl_usd
    burn_activity = activity[activity.event_type == "burn"]
    same_slot = latencies[latencies.latency_slots == 0]
    summary: dict[str, object] = {
        "scope": {
            "activity_rows": int(len(activity)),
            "activity_transactions": int(activity.tx_hash.nunique()),
            "wallet_bought_tokens": int(first_buys.token_address.nunique()),
            "core_bought_deployment_tokens": int(len(latencies)),
            "core_tokens_with_first_buy": int(latencies.buy_tx_hash.notna().sum()),
        },
        "entry_usd_core": _quantiles(latencies.entry_cost_usd),
        "entry_quote_core": _quantiles(latencies.entry_quote_amount),
        "latency_seconds": _quantiles(latencies.latency_seconds),
        "latency_slots": _quantiles(latencies.latency_slots),
        "zero_slot": {
            "count": int(latencies.latency_slots.eq(0).sum()),
            "share": float(latencies.latency_slots.eq(0).mean()),
        },
        "same_slot_position": {
            "observations": int(same_slot.same_slot_tx_delta.notna().sum()),
            "next_transaction_count": int(same_slot.same_slot_tx_delta.eq(1).sum()),
            "next_transaction_share": float(same_slot.same_slot_tx_delta.eq(1).mean()),
            "median_tx_delta": float(same_slot.same_slot_tx_delta.median()),
            "negative_delta_count": int(same_slot.same_slot_tx_delta.lt(0).sum()),
        },
        "june_same_bundle": {
            "paired_tokens": int(same_bundle.token_address.nunique()) if not same_bundle.empty else 0,
            "next_bundle_transaction_count": int(same_bundle.bundle_index_delta.eq(1).sum()) if not same_bundle.empty else 0,
            "median_bundle_delta": float(same_bundle.bundle_index_delta.median()) if not same_bundle.empty else None,
        },
        "hold_seconds": _quantiles(bought_positions.hold_seconds),
        "exit_structure": {
            "bought_position_tokens": int(len(bought_positions)),
            "tokens_with_partial_exits": int(bought_positions.partial_sell_transactions.gt(0).sum()),
            "partial_exit_share": float(bought_positions.partial_sell_transactions.gt(0).mean()),
            "sell_transactions_for_bought_positions": int(bought_positions.sell_transactions.sum()),
            "all_wallet_sell_transactions": int(activity.event_type.eq("sell").sum()),
            "mean_sells_per_bought_token": float(bought_positions.sell_transactions.mean()),
            "median_sells_per_bought_token": float(bought_positions.sell_transactions.median()),
            "tokens_with_burn_activity": int(burn_activity.token_address.nunique()),
            "burn_transactions": int(burn_activity.tx_hash.nunique()),
        },
        "cashflow_performance_bought_positions": {
            "scope_bought_positions": int(len(bought_positions)),
            "quote_principal_buy_usd": float(bought_positions.gross_buy_usd.sum()),
            "quote_principal_sell_usd": float(bought_positions.gross_sell_usd.sum()),
            "network_execution_cost_usd": float(bought_positions.network_cost_usd.sum()),
            "priority_fee_included_native": float(bought_positions.priority_fee_included_native.sum()),
            "tip_fee_included_native": float(bought_positions.tip_fee_included_native.sum()),
            "pump_target_route_rate": PUMP_TARGET_ROUTE_TOTAL_RATE,
            "pump_separate_cost_usd": float(bought_positions.pump_separate_cost_usd.sum()),
            "pump_reported_component_usd": float(bought_positions.pump_reported_component_usd.sum()),
            "pump_raw_observed_fallback_usd": float(bought_positions.pump_raw_observed_fallback_usd.sum()),
            "routed_dex_contained_in_quote_usd_not_subtracted": float(bought_positions.routed_dex_contained_in_quote_usd.sum()),
            "unidentified_dex_ambiguous_usd_not_subtracted": float(bought_positions.unidentified_dex_ambiguous_usd.sum()),
            "pump_reported_residual_ambiguous_usd_not_subtracted": float(bought_positions.pump_reported_residual_ambiguous_usd.sum()),
            "total_defensible_cost_usd": float(bought_positions.total_defensible_cost_usd.sum()),
            "gross_pnl_usd": float(bought_positions.gross_pnl_usd.sum()),
            "fully_fee_adjusted_pnl_usd": float(bought_positions.net_pnl_usd.sum()),
            "fully_fee_adjusted_hit_rate": float(bought_positions.net_pnl_usd.gt(0).mean()),
            "average_winner_usd": float(winning.mean()),
            "average_loser_usd": float(losing.mean()),
            "median_fully_fee_adjusted_pnl_usd": float(bought_positions.net_pnl_usd.median()),
            "mean_fully_fee_adjusted_pnl_usd": float(bought_positions.net_pnl_usd.mean()),
        },
        "limitations": [
            "P&L is quote-principal cash flow less inclusive transaction gas and the raw-supported 1.25% Pump target-route cost; it is not a full mark-to-market ledger.",
            "Priority and tip components are already inside gas and are never subtracted again.",
            "The additional Pump amount needed beyond reported dex_* to reach the raw-observed target-route total is intentionally not assigned an unproved semantic fee name.",
            "Known routed-venue dex_* fields are not charged again because the observed quote transfer is already net; blank-venue DEX fields and rent/residual-account movements remain ambiguous.",
            "Same-slot transaction indexes establish relative landed order for the two known transactions but not private mempool observability.",
            "Same-bundle analysis is June-only and ex-post behavioral context; Jito fields are not model features.",
        ],
    }
    summary_path = SUBMISSION / "tables" / "behavior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    exit_table = token_metrics[
        [
            "sell_transactions",
            "partial_sell_transactions",
            "close_sell_transactions",
            "burn_transactions",
        ]
    ].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    exit_table.to_csv(SUBMISSION / "tables" / "behavior_exit_summary.csv")
    _figures(core_metrics, bought_positions, monthly)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct target-wallet behavior")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
