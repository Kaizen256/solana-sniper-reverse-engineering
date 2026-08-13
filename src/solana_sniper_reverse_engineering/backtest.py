from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

from .behavior import LATENCIES, TOKEN_METRICS
from .config import ARTIFACTS, JUNE_START, JUNE_TRADES, SUBMISSION, TARGET_ACTIVITY, ensure_output_dirs
from .modeling import PREDICTIONS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTCOMES = ARTIFACTS / "tables" / "june_strategy_outcomes.parquet"
RESULTS = SUBMISSION / "tables" / "backtest_results.json"
INTRASLOT_OUTCOMES = ARTIFACTS / "tables" / "june_intraslot_outcomes.parquet"
INTRASLOT_RESULTS = SUBMISSION / "tables" / "intraslot_latency_sensitivity.json"
INTRASLOT_TABLE = SUBMISSION / "tables" / "intraslot_latency_sensitivity.csv"
INTRASLOT_OFFSETS = (1, 10, 25, 50, 100, 118, 150, 250)


def _strategy_parameters() -> tuple[float, int, float]:
    latencies = pd.read_parquet(LATENCIES)
    pre_june = latencies[
        (latencies.deploy_time < JUNE_START)
        & latencies.quote_token_symbol.isin(["SOL", "WSOL"])
        & latencies.entry_quote_amount.gt(0)
    ]
    stake_sol = float(pre_june.entry_quote_amount.median())

    activity = pd.read_parquet(
        TARGET_ACTIVITY,
        columns=["token_address", "timestamp", "event_type", "is_open_or_close", "tx_hash", "gas_native"],
    )
    activity["gas_native"] = pd.to_numeric(activity.gas_native, errors="coerce").fillna(0)
    pre = activity[activity.timestamp < JUNE_START].copy()
    buys = (
        pre[pre.event_type == "buy"]
        .sort_values(["timestamp", "tx_hash"])
        .drop_duplicates("token_address")
        [["token_address", "timestamp", "gas_native"]]
        .rename(columns={"timestamp": "buy_time", "gas_native": "buy_fee"})
    )
    closes = (
        pre[(pre.event_type == "sell") & (pre.is_open_or_close == 1)]
        .sort_values(["timestamp", "tx_hash"])
        .drop_duplicates("token_address", keep="last")
        [["token_address", "timestamp", "gas_native"]]
        .rename(columns={"timestamp": "close_time", "gas_native": "close_fee"})
    )
    round_trips = buys.merge(closes, on="token_address", how="inner")
    hold_seconds = int(np.nanmedian(round_trips.close_time - round_trips.buy_time))
    fee_sol = float(np.nanmedian(round_trips.buy_fee + round_trips.close_fee))
    return stake_sol, hold_seconds, fee_sol


def _build_outcomes(predictions: pd.DataFrame, hold_seconds: int, force: bool) -> pd.DataFrame:
    if OUTCOMES.exists() and not force:
        return pd.read_parquet(OUTCOMES)
    wanted = predictions[(predictions.selected == 1) | (predictions.label == 1)][
        ["token_address", "block_time", "label", "selected", "score"]
    ].copy()
    con = duckdb.connect()
    con.execute("SET memory_limit='36GB'")
    con.execute("SET threads=20")
    con.execute("SET preserve_insertion_order=false")
    temp = ARTIFACTS / "duckdb_temp"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp.as_posix()}'")
    con.register("wanted", wanted)
    result = con.execute(
        f"""
        WITH filtered_trades AS MATERIALIZED (
          SELECT t.token_address, t.block_slot, t.tx_index, t.event_index,
                 t.block_time, t.price_sol, t.side,
                 t.deploy_block_slot, t.deploy_tx_index
          FROM read_parquet('{JUNE_TRADES.as_posix()}') t
          SEMI JOIN wanted w USING (token_address)
          WHERE t.price_sol > 0
        ), delays(delay_slots) AS (VALUES (0), (1), (2)),
        entry_candidates AS (
          SELECT w.*, d.delay_slots, t.block_slot AS entry_slot,
                 t.tx_index AS entry_tx_index, t.event_index AS entry_event_index,
                 t.block_time AS entry_time, t.price_sol AS entry_price_sol,
                 row_number() OVER (
                   PARTITION BY w.token_address, d.delay_slots
                   ORDER BY t.block_slot, t.tx_index, t.event_index
                 ) AS rn
          FROM wanted w CROSS JOIN delays d
          JOIN filtered_trades t USING (token_address)
          WHERE (d.delay_slots=0 AND (
                   t.block_slot > t.deploy_block_slot OR
                   (t.block_slot=t.deploy_block_slot AND t.tx_index>t.deploy_tx_index)
                 ))
             OR (d.delay_slots>0 AND t.block_slot>=t.deploy_block_slot+d.delay_slots)
        ), entries AS (
          SELECT * EXCLUDE (rn) FROM entry_candidates WHERE rn=1
        ), exit_candidates AS (
          SELECT e.*, t.block_slot AS exit_slot, t.tx_index AS exit_tx_index,
                 t.event_index AS exit_event_index, t.block_time AS exit_time,
                 t.price_sol AS exit_price_sol,
                 row_number() OVER (
                   PARTITION BY e.token_address, e.delay_slots
                   ORDER BY t.block_slot, t.tx_index, t.event_index
                 ) AS rn
          FROM entries e
          JOIN filtered_trades t USING (token_address)
          WHERE t.block_time >= e.entry_time + {int(hold_seconds)}
        ), exits AS (
          SELECT * EXCLUDE (rn) FROM exit_candidates WHERE rn=1
        )
        SELECT e.*, x.exit_slot, x.exit_tx_index, x.exit_event_index,
               x.exit_time, x.exit_price_sol,
               CASE WHEN x.exit_price_sol IS NULL THEN -1.0
                    ELSE x.exit_price_sol/e.entry_price_sol-1.0 END AS gross_roi,
               (x.exit_price_sol IS NULL)::UTINYINT AS forced_total_loss
        FROM entries e
        LEFT JOIN exits x USING (
          token_address, block_time, label, selected, score, delay_slots,
          entry_slot, entry_tx_index, entry_event_index, entry_time, entry_price_sol
        )
        """
    ).fetch_df()
    result.to_parquet(OUTCOMES, index=False)
    return result


def _empirical_same_slot_offsets(tokens: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    """Assign deterministic offsets from the pre-June target same-slot distribution.

    The token hash controls the empirical draw, so the latency assignment is independent
    of labels, predictions, and June outcomes.
    """
    latencies = pd.read_parquet(
        LATENCIES,
        columns=["deploy_time", "latency_slots", "same_slot_tx_delta"],
    )
    pool = (
        latencies[
            (latencies.deploy_time < JUNE_START)
            & latencies.latency_slots.eq(0)
            & latencies.same_slot_tx_delta.ge(1)
        ]
        .same_slot_tx_delta.dropna()
        .astype("int64")
        .to_numpy()
    )
    if not len(pool):
        raise RuntimeError("No positive pre-June same-slot transaction offsets available")

    def draw(token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int(pool[int.from_bytes(digest[:8], "big") % len(pool)])

    assigned = tokens.astype(str).map(draw).to_numpy(dtype=np.int64)
    summary: dict[str, object] = {
        "source": "pre-June target entries with latency_slots=0 and same_slot_tx_delta>=1",
        "observations": int(len(pool)),
        "assignment": "SHA-256(token_address) indexes the empirical pool; independent of labels and outcomes",
        "conditional_scope": "same-slot target entries only; this is not a wall-clock latency distribution",
        "min": int(pool.min()),
        "p10": float(np.quantile(pool, 0.10)),
        "median": float(np.median(pool)),
        "p90": float(np.quantile(pool, 0.90)),
        "p99": float(np.quantile(pool, 0.99)),
        "max": int(pool.max()),
    }
    return assigned, summary


def _build_intraslot_outcomes(
    predictions: pd.DataFrame,
    hold_seconds: int,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate transaction-position latency using only June trade ordering fields.

    Entry must occur inside the deployment slot at a transaction index at least the
    requested offset after ``deploy_tx_index``. A token without such a trade is a
    no-fill. This deliberately makes no wall-clock, mempool, or private-ordering claim.
    """
    selection_columns = [
        column
        for column in ("selected", "active_era_selected", "legacy_population_selected")
        if column in predictions
    ]
    wanted_mask = predictions.label.eq(1)
    for column in selection_columns:
        wanted_mask |= predictions[column].eq(1)
    columns = [
        "token_address",
        "block_time",
        "label",
        "score",
        *selection_columns,
    ]
    wanted = predictions.loc[wanted_mask, columns].copy()
    empirical, empirical_summary = _empirical_same_slot_offsets(wanted.token_address)

    if INTRASLOT_OUTCOMES.exists() and not force:
        return pd.read_parquet(INTRASLOT_OUTCOMES), empirical_summary

    requirement_frames = []
    for offset in INTRASLOT_OFFSETS:
        frame = wanted.copy()
        frame["latency_policy"] = f"offset_{offset}"
        frame["tx_offset"] = offset
        requirement_frames.append(frame)
    empirical_frame = wanted.copy()
    empirical_frame["latency_policy"] = "empirical_pre_june_same_slot"
    empirical_frame["tx_offset"] = empirical
    requirement_frames.append(empirical_frame)
    requirements = pd.concat(requirement_frames, ignore_index=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='36GB'")
    con.execute("SET threads=20")
    con.execute("SET preserve_insertion_order=false")
    temp = ARTIFACTS / "duckdb_temp_intraslot"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp.as_posix()}'")
    con.register("requirements", requirements)
    carry = ", ".join(f"r.{column}" for column in selection_columns)
    if carry:
        carry = ", " + carry
    result = con.execute(
        f"""
        WITH filtered_trades AS MATERIALIZED (
          SELECT t.token_address, t.block_slot, t.tx_index, t.event_index,
                 t.block_time, t.price_sol, t.deploy_block_slot, t.deploy_tx_index
          FROM read_parquet('{JUNE_TRADES.as_posix()}') t
          SEMI JOIN (SELECT DISTINCT token_address FROM requirements) w USING (token_address)
          WHERE t.price_sol > 0
        ), entry_candidates AS (
          SELECT r.token_address, r.block_time AS deploy_time, r.label, r.score
                 {carry}, r.latency_policy, r.tx_offset,
                 t.deploy_block_slot, t.deploy_tx_index,
                 t.block_slot AS entry_slot, t.tx_index AS entry_tx_index,
                 t.event_index AS entry_event_index, t.block_time AS entry_time,
                 t.price_sol AS entry_price_sol,
                 row_number() OVER (
                   PARTITION BY r.token_address, r.latency_policy
                   ORDER BY t.tx_index, t.event_index
                 ) AS rn
          FROM requirements r
          JOIN filtered_trades t USING (token_address)
          WHERE t.block_slot = t.deploy_block_slot
            AND t.tx_index >= t.deploy_tx_index + r.tx_offset
        ), entries AS (
          SELECT * EXCLUDE (rn) FROM entry_candidates WHERE rn=1
        ), exit_candidates AS (
          SELECT e.token_address, e.latency_policy,
                 t.block_slot AS exit_slot, t.tx_index AS exit_tx_index,
                 t.event_index AS exit_event_index, t.block_time AS exit_time,
                 t.price_sol AS exit_price_sol,
                 row_number() OVER (
                   PARTITION BY e.token_address, e.latency_policy
                   ORDER BY t.block_slot, t.tx_index, t.event_index
                 ) AS rn
          FROM entries e
          JOIN filtered_trades t USING (token_address)
          WHERE t.block_time >= e.entry_time + {int(hold_seconds)}
        ), exits AS (
          SELECT * EXCLUDE (rn) FROM exit_candidates WHERE rn=1
        )
        SELECT e.*, x.exit_slot, x.exit_tx_index, x.exit_event_index,
               x.exit_time, x.exit_price_sol,
               CASE WHEN x.exit_price_sol IS NULL THEN -1.0
                    ELSE x.exit_price_sol/e.entry_price_sol-1.0 END AS gross_roi,
               (x.exit_price_sol IS NULL)::UTINYINT AS forced_total_loss
        FROM entries e
        LEFT JOIN exits x USING (token_address, latency_policy)
        """
    ).fetch_df()
    result.to_parquet(INTRASLOT_OUTCOMES, index=False)
    return result, empirical_summary


def run_intraslot_sensitivity(
    predictions: pd.DataFrame | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run fixed and empirical transaction-index sensitivity for June cohorts."""
    ensure_output_dirs()
    if predictions is None:
        predictions = pd.read_parquet(PREDICTIONS)
        predictions = predictions[predictions.split == "test"].copy()
    stake_sol, hold_seconds, fee_sol = _strategy_parameters()
    outcomes, empirical_summary = _build_intraslot_outcomes(
        predictions, hold_seconds, force=force
    )
    cohort_columns = {"replica": "selected", "target_equal_stake": "label"}
    if "active_era_selected" in predictions and not predictions[
        "active_era_selected"
    ].equals(predictions["selected"]):
        cohort_columns["active_era_replica_diagnostic"] = "active_era_selected"
    if "legacy_population_selected" in predictions:
        cohort_columns["legacy_population_replica_diagnostic"] = (
            "legacy_population_selected"
        )

    results: dict[str, object] = {}
    flat_rows: list[dict[str, object]] = []
    policies = [f"offset_{offset}" for offset in INTRASLOT_OFFSETS] + [
        "empirical_pre_june_same_slot"
    ]
    for policy in policies:
        policy_rows = outcomes[outcomes.latency_policy == policy]
        policy_result: dict[str, object] = {}
        for cohort, column in cohort_columns.items():
            expected = int(predictions[column].sum())
            values, _ = _metrics(
                policy_rows[policy_rows[column] == 1], expected, stake_sol, fee_sol
            )
            policy_result[cohort] = values
            flat_rows.append(
                {
                    "latency_policy": policy,
                    "fixed_tx_offset": (
                        int(policy.removeprefix("offset_"))
                        if policy.startswith("offset_")
                        else None
                    ),
                    "cohort": cohort,
                    **values,
                }
            )
        results[policy] = policy_result
    pd.DataFrame(flat_rows).to_csv(INTRASLOT_TABLE, index=False)
    output: dict[str, object] = {
        "entry_rule": "first positive-price June trade in the deployment slot with tx_index >= deploy_tx_index + offset",
        "no_fill_rule": "no eligible trade remaining in the deployment slot means no fill; execution is not rolled into a later slot",
        "exit_and_cost": {
            "hold_seconds": hold_seconds,
            "stake_sol": stake_sol,
            "round_trip_fee_sol": fee_sol,
            "missing_exit": "-100% gross return",
        },
        "empirical_policy": empirical_summary,
        "results": results,
        "limitations": [
            "tx_index is landed transaction order, not elapsed wall-clock latency.",
            "The trade table establishes ordering only for recorded Pump.fun trades; it does not reveal private submission paths, mempool visibility, or bundle intent.",
            "Event prices are marginal observed prices and omit the strategy's own bonding-curve impact.",
            "The empirical policy is conditional on pre-June target entries that were same-slot; it does not model the target's non-same-slot tail.",
        ],
    }
    INTRASLOT_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def _metrics(
    cohort: pd.DataFrame,
    expected_tokens: int,
    stake_sol: float,
    fee_sol: float,
    symmetric_slippage: float = 0.0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    ordered = cohort.sort_values(["entry_time", "token_address"]).copy()
    price_ratio = (1.0 + ordered.gross_roi) * (1.0 - symmetric_slippage) / (
        1.0 + symmetric_slippage
    )
    ordered["net_roi"] = np.where(
        ordered.forced_total_loss.eq(1),
        -1.0 - fee_sol / stake_sol,
        price_ratio - 1.0 - fee_sol / stake_sol,
    )
    ordered["pnl_sol"] = ordered.net_roi * stake_sol
    ordered["equity_sol"] = ordered.pnl_sol.cumsum()
    running_peak = np.maximum.accumulate(np.r_[0.0, ordered.equity_sol.to_numpy()])
    drawdowns = running_peak[1:] - ordered.equity_sol.to_numpy()
    positive_pnl = ordered.pnl_sol.clip(lower=0)
    total_positive = positive_pnl.sum()
    p99 = ordered.net_roi.quantile(0.99) if len(ordered) else 0.0
    values: dict[str, float | int] = {
        "selected_tokens": int(expected_tokens),
        "filled_trades": int(len(ordered)),
        "fill_rate": float(len(ordered) / expected_tokens) if expected_tokens else 0.0,
        "forced_total_losses": int(ordered.forced_total_loss.sum()),
        "hit_rate": float(ordered.net_roi.gt(0).mean()) if len(ordered) else 0.0,
        "mean_roi": float(ordered.net_roi.mean()) if len(ordered) else 0.0,
        "median_roi": float(ordered.net_roi.median()) if len(ordered) else 0.0,
        "p90_roi": float(ordered.net_roi.quantile(0.90)) if len(ordered) else 0.0,
        "p99_roi": float(p99),
        "max_roi": float(ordered.net_roi.max()) if len(ordered) else 0.0,
        "total_pnl_sol": float(ordered.pnl_sol.sum()),
        "total_pnl_sol_roi_capped_at_p99": float(
            (ordered.net_roi.clip(upper=p99) * stake_sol).sum()
        ) if len(ordered) else 0.0,
        "top10_positive_pnl_share": float(positive_pnl.nlargest(10).sum() / total_positive)
        if total_positive > 0
        else 0.0,
        "roi_on_total_staked": float(ordered.pnl_sol.sum() / (stake_sol * len(ordered))) if len(ordered) else 0.0,
        "max_drawdown_sol": float(drawdowns.max()) if drawdowns.size else 0.0,
        "symmetric_entry_exit_slippage": symmetric_slippage,
    }
    return values, ordered


def _actual_target_june() -> dict[str, float | int]:
    latencies = pd.read_parquet(LATENCIES, columns=["token_address", "deploy_time"])
    metrics = pd.read_parquet(TOKEN_METRICS)
    june = latencies[latencies.deploy_time >= JUNE_START].merge(metrics, on="token_address", how="left")
    invested = float(june.gross_buy_usd.sum() + june.fees_usd.sum())
    return {
        "tokens": int(len(june)),
        "hit_rate_net_cashflow": float(june.net_pnl_usd.gt(0).mean()),
        "gross_buy_usd": float(june.gross_buy_usd.sum()),
        "fees_usd": float(june.fees_usd.sum()),
        "net_pnl_usd": float(june.net_pnl_usd.sum()),
        "net_roi_on_buy_plus_fees": float(june.net_pnl_usd.sum() / invested),
        "mean_net_pnl_usd": float(june.net_pnl_usd.mean()),
        "median_net_pnl_usd": float(june.net_pnl_usd.median()),
    }


def _plot(equities: dict[str, pd.DataFrame]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    colors = {0: "#176B87", 1: "#E07A2D", 2: "#7A5195"}
    for delay in (0, 1, 2):
        replica = equities[f"replica_{delay}"]
        axes[0].plot(
            np.arange(len(replica)),
            replica.equity_sol,
            label=f"preserved control +{delay} slot",
            color=colors[delay],
        )
    target = equities["target_0"]
    axes[0].plot(
        np.arange(len(target)),
        target.equity_sol,
        label="target cohort +0 slot",
        color="black",
        alpha=0.7,
    )
    axes[0].set(
        xlabel="Executed trades (chronological)",
        ylabel="Cumulative P&L (SOL)",
        title="Optimistic marginal equal-stake curves",
    )
    axes[0].legend(fontsize=8)

    distributions = [equities[f"replica_{delay}"].net_roi.clip(-1.1, 3) for delay in (0, 1, 2)]
    axes[1].boxplot(distributions, tick_labels=["+0", "+1", "+2"], showfliers=False)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        xlabel="Entry delay (slots)",
        ylabel="Net ROI",
        title="Preserved-control delay sensitivity",
    )
    fig.tight_layout()
    for directory in (ARTIFACTS / "figures", SUBMISSION / "figures"):
        fig.savefig(directory / "06_backtest_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(force: bool = False) -> dict[str, object]:
    ensure_output_dirs()
    predictions = pd.read_parquet(PREDICTIONS)
    predictions = predictions[predictions.split == "test"].copy()
    stake_sol, hold_seconds, primary_fee_sol = _strategy_parameters()
    outcomes = _build_outcomes(predictions, hold_seconds, force)
    expected_replica = int(predictions.selected.sum())
    expected_target = int(predictions.label.sum())
    output: dict[str, object] = {
        "strategy": {
            "selection": "fixed May-validation score threshold",
            "entry": "first observed positive-price trade strictly after deploy tx (+0), or at/after deploy slot +1/+2",
            "exit": f"first observed trade at least {hold_seconds}s after entry; horizon fixed from pre-June target median hold",
            "stake_sol": stake_sol,
            "primary_round_trip_fee_sol": primary_fee_sol,
            "missing_exit": "-100% gross return",
            "limitations": [
                "The zero-slippage result is an optimistic marginal-price upper bound; observed event prices do not model our own bonding-curve price impact.",
                "A future observed trade supplies the exit price; tokens with no such trade are total losses.",
                "No-fill tokens are reported in fill rates and are not silently assigned favorable prices.",
                "Symmetric 10%/25%/50% adverse entry-and-exit price sensitivities bound unsupported fill assumptions.",
            ],
        },
        "selection_overlap": {
            "replica_entries": expected_replica,
            "target_entries": expected_target,
            "overlap": int(((predictions.selected == 1) & (predictions.label == 1)).sum()),
            "precision_vs_target": float(predictions.loc[predictions.selected == 1, "label"].mean()) if expected_replica else 0.0,
            "recall_of_target": float(predictions.loc[predictions.label == 1, "selected"].mean()) if expected_target else 0.0,
        },
        "primary_fee_results": {},
        "fee_sensitivity_replica_delay0": {},
        "slippage_sensitivity_delay0": {},
        "actual_target_cashflow_june": _actual_target_june(),
    }
    equities: dict[str, pd.DataFrame] = {}
    for delay in (0, 1, 2):
        delayed = outcomes[outcomes.delay_slots == delay]
        replica = delayed[delayed.selected == 1]
        target = delayed[delayed.label == 1]
        replica_metrics, replica_equity = _metrics(replica, expected_replica, stake_sol, primary_fee_sol)
        target_metrics, target_equity = _metrics(target, expected_target, stake_sol, primary_fee_sol)
        output["primary_fee_results"][f"delay_{delay}"] = {  # type: ignore[index]
            "replica": replica_metrics,
            "target_equal_stake": target_metrics,
        }
        equities[f"replica_{delay}"] = replica_equity
        equities[f"target_{delay}"] = target_equity
    for fee in sorted(set([0.0, 0.01, round(primary_fee_sol, 6), 0.03, 0.05])):
        values, _ = _metrics(
            outcomes[(outcomes.delay_slots == 0) & (outcomes.selected == 1)],
            expected_replica,
            stake_sol,
            fee,
        )
        output["fee_sensitivity_replica_delay0"][f"{fee:.6f}_sol"] = values  # type: ignore[index]
    for slippage in (0.0, 0.10, 0.25, 0.50):
        replica_values, _ = _metrics(
            outcomes[(outcomes.delay_slots == 0) & (outcomes.selected == 1)],
            expected_replica,
            stake_sol,
            primary_fee_sol,
            symmetric_slippage=slippage,
        )
        target_values, _ = _metrics(
            outcomes[(outcomes.delay_slots == 0) & (outcomes.label == 1)],
            expected_target,
            stake_sol,
            primary_fee_sol,
            symmetric_slippage=slippage,
        )
        output["slippage_sensitivity_delay0"][f"{slippage:.2f}"] = {  # type: ignore[index]
            "replica": replica_values,
            "target_equal_stake": target_values,
        }

    delay0 = outcomes[outcomes.delay_slots == 0].copy()
    delay0["net_roi"] = delay0.gross_roi - primary_fee_sol / stake_sol
    disagreement = delay0[
        ((delay0.selected == 1) & (delay0.label == 0))
        | ((delay0.selected == 0) & (delay0.label == 1))
    ].copy()
    disagreement["disagreement"] = np.where(
        disagreement.selected.eq(1), "replica_only", "target_only"
    )
    disagreement.sort_values("net_roi").to_parquet(
        ARTIFACTS / "tables" / "strategy_disagreements.parquet", index=False
    )
    (
        disagreement.assign(profitable=disagreement.net_roi.gt(0))
        .groupby(["disagreement", "profitable"], as_index=False)
        .agg(cases=("token_address", "size"), mean_net_roi=("net_roi", "mean"), median_net_roi=("net_roi", "median"))
        .to_csv(SUBMISSION / "tables" / "strategy_disagreement_summary.csv", index=False)
    )
    output["disagreements"] = {
        "replica_only_profitable": int(
            ((disagreement.disagreement == "replica_only") & (disagreement.net_roi > 0)).sum()
        ),
        "replica_only_harmful": int(
            ((disagreement.disagreement == "replica_only") & (disagreement.net_roi <= 0)).sum()
        ),
        "missed_target_profitable": int(
            ((disagreement.disagreement == "target_only") & (disagreement.net_roi > 0)).sum()
        ),
        "missed_target_harmful": int(
            ((disagreement.disagreement == "target_only") & (disagreement.net_roi <= 0)).sum()
        ),
    }
    RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _plot(equities)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen June replica/backtest comparison")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
