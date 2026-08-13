from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import _strategy_parameters
from .config import ARTIFACTS, JUNE_TRADES, SUBMISSION, ensure_output_dirs
from .third_pass import _connection, _sql


LAMPORTS_PER_SOL = 1_000_000_000
BASE_UNITS_PER_TOKEN = 1_000_000
INITIAL_VIRTUAL_TOKEN_RESERVES = 1_073_000_000_000_000
INITIAL_VIRTUAL_SOL_RESERVES = 30_000_000_000
INITIAL_REAL_TOKEN_RESERVES = 793_100_000_000_000
INITIAL_REAL_SOL_RESERVES = 0

FEE_RATE_LOWER = 0.0095
FEE_RATE_UPPER = 0.0125

STRATEGY_PREDICTIONS = ARTIFACTS / "tables" / "third_pass_june_strategy_predictions.parquet"
CURVE_OUTCOMES = ARTIFACTS / "tables" / "june_curve_replay_outcomes.parquet"
CURVE_RESULTS = SUBMISSION / "tables" / "curve_replay_results.json"
CURVE_COVERAGE = SUBMISSION / "tables" / "curve_replay_coverage.csv"


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


@dataclass(frozen=True)
class CurveState:
    virtual_token_reserves: int = INITIAL_VIRTUAL_TOKEN_RESERVES
    virtual_sol_reserves: int = INITIAL_VIRTUAL_SOL_RESERVES
    real_token_reserves: int = INITIAL_REAL_TOKEN_RESERVES
    real_sol_reserves: int = INITIAL_REAL_SOL_RESERVES

    @property
    def price_sol(self) -> float:
        return (
            self.virtual_sol_reserves
            / self.virtual_token_reserves
            * BASE_UNITS_PER_TOKEN
            / LAMPORTS_PER_SOL
        )

    def buy_exact_net_sol(self, net_sol: int) -> tuple[CurveState, int]:
        """Apply Pump's integer SOL-in quote after fees have been removed."""
        if net_sol <= 1:
            raise ValueError("net SOL input must exceed one lamport")
        token_out = (
            (net_sol - 1) * self.virtual_token_reserves
            // (self.virtual_sol_reserves + net_sol - 1)
        )
        if token_out <= 0 or token_out > self.real_token_reserves:
            raise ValueError("buy would exhaust or exceed real token reserves")
        return self.apply_buy(token_out, net_sol), token_out

    def buy_exact_tokens(self, token_out: int) -> tuple[CurveState, int]:
        """Apply the official reverse quote for a fixed token output."""
        if token_out <= 0 or token_out >= self.virtual_token_reserves:
            raise ValueError("invalid token output")
        if token_out > self.real_token_reserves:
            raise ValueError("buy would exceed real token reserves")
        net_sol = (
            _ceil_div(
                token_out * self.virtual_sol_reserves,
                self.virtual_token_reserves - token_out,
            )
            + 1
        )
        return self.apply_buy(token_out, net_sol), net_sol

    def sell_exact_tokens(self, token_in: int) -> tuple[CurveState, int]:
        """Sell a fixed token input against the constant-product curve, before fees."""
        if token_in <= 0:
            raise ValueError("token input must be positive")
        gross_sol = (
            token_in
            * self.virtual_sol_reserves
            // (self.virtual_token_reserves + token_in)
        )
        if gross_sol <= 0 or gross_sol > self.real_sol_reserves:
            raise ValueError("sell exceeds real SOL reserves")
        return self.apply_sell(token_in, gross_sol), gross_sol

    def apply_buy(self, token_out: int, net_sol: int) -> CurveState:
        if token_out <= 0 or net_sol <= 0:
            raise ValueError("buy deltas must be positive")
        if token_out > self.real_token_reserves:
            raise ValueError("buy exceeds real token reserves")
        return replace(
            self,
            virtual_token_reserves=self.virtual_token_reserves - token_out,
            virtual_sol_reserves=self.virtual_sol_reserves + net_sol,
            real_token_reserves=self.real_token_reserves - token_out,
            real_sol_reserves=self.real_sol_reserves + net_sol,
        )

    def apply_sell(self, token_in: int, gross_sol: int) -> CurveState:
        if token_in <= 0 or gross_sol <= 0:
            raise ValueError("sell deltas must be positive")
        if gross_sol > self.real_sol_reserves:
            raise ValueError("sell exceeds real SOL reserves")
        return replace(
            self,
            virtual_token_reserves=self.virtual_token_reserves + token_in,
            virtual_sol_reserves=self.virtual_sol_reserves - gross_sol,
            real_token_reserves=self.real_token_reserves + token_in,
            real_sol_reserves=self.real_sol_reserves - gross_sol,
        )


def _to_lamports(value: float) -> int:
    return int(round(float(value) * LAMPORTS_PER_SOL))


def _to_base_units(value: float) -> int:
    return int(round(float(value) * BASE_UNITS_PER_TOKEN))


def strategy_cashflow(
    net_entry_sol: float,
    gross_exit_sol: float,
    network_fee_sol: float,
    fee_rate: float,
) -> tuple[float, float, float]:
    """Return gross entry spend, net exit receipt, and P&L after both swap fees."""
    if min(net_entry_sol, gross_exit_sol, network_fee_sol, fee_rate) < 0:
        raise ValueError("cashflow inputs must be nonnegative")
    entry_gross_sol = net_entry_sol * (1.0 + fee_rate)
    exit_net_sol = gross_exit_sol * (1.0 - fee_rate)
    pnl_sol = exit_net_sol - entry_gross_sol - network_fee_sol
    return entry_gross_sol, exit_net_sol, pnl_sol


def _position(row: pd.Series) -> tuple[int, int, int]:
    return int(row.block_slot), int(row.tx_index), int(row.event_index)


def _standard_curve_compatible(first: pd.Series) -> tuple[bool, dict[str, float]]:
    if first.side != "buy" or first.program != "pump":
        return False, {
            "normalized_quote_error_sol": float("nan"),
            "price_relative_error": float("nan"),
        }
    state = CurveState()
    recorded_tokens = _to_base_units(first.base_amount)
    try:
        post, inferred_net_sol = state.buy_exact_tokens(recorded_tokens)
    except ValueError:
        return False, {
            "normalized_quote_error_sol": float("inf"),
            "price_relative_error": float("inf"),
        }
    normalized_quote_error_sol = abs(
        inferred_net_sol - _to_lamports(first.amount_sol)
    ) / LAMPORTS_PER_SOL
    price_relative_error = abs(post.price_sol - float(first.price_sol)) / float(first.price_sol)
    # Price is derived from the emitted post-trade reserves and is the invariant.
    # Normalized quote amounts are retained only as a decoder-quality diagnostic.
    compatible = price_relative_error <= 1e-6
    return compatible, {
        "normalized_quote_error_sol": normalized_quote_error_sol,
        "price_relative_error": price_relative_error,
    }


def _apply_recorded(state: CurveState, row: pd.Series) -> CurveState:
    token_amount = _to_base_units(row.base_amount)
    sol_amount = _to_lamports(row.amount_sol)
    if row.side == "buy":
        return state.apply_buy(token_amount, sol_amount)
    if row.side == "sell":
        return state.apply_sell(token_amount, sol_amount)
    raise ValueError(f"unsupported side {row.side!r}")


def _apply_counterfactual_market_trade(
    state: CurveState,
    row: pd.Series,
    buy_intent: str,
    observed_net_sol: int,
) -> CurveState:
    token_amount = _to_base_units(row.base_amount)
    if row.side == "sell":
        return state.sell_exact_tokens(token_amount)[0]
    if row.side != "buy":
        raise ValueError(f"unsupported side {row.side!r}")
    if buy_intent == "fixed_quote":
        return state.buy_exact_net_sol(observed_net_sol)[0]
    if buy_intent == "fixed_token":
        return state.buy_exact_tokens(token_amount)[0]
    raise ValueError(f"unsupported buy intent {buy_intent!r}")


def replay_one(
    trades: pd.DataFrame,
    anchor: pd.Series,
    net_stake_sol: float,
    network_fee_sol: float,
    buy_intent: str,
    fee_rate: float,
) -> dict[str, object]:
    """Insert our buy after the entry anchor and our sell after the exit anchor."""
    entry_position = (
        int(anchor.entry_slot),
        int(anchor.entry_tx_index),
        int(anchor.entry_event_index),
    )
    exit_position = (
        int(anchor.exit_slot),
        int(anchor.exit_tx_index),
        int(anchor.exit_event_index),
    )
    ordered = trades.sort_values(["block_slot", "tx_index", "event_index"])
    first = ordered.iloc[0]
    compatible, compatibility = _standard_curve_compatible(first)
    base: dict[str, object] = {
        "token_address": anchor.token_address,
        "latency_policy": anchor.latency_policy,
        "buy_intent_bound": buy_intent,
        "fee_rate": fee_rate,
        "status": "ok",
        **compatibility,
    }
    if not compatible:
        return {**base, "status": "nonstandard_or_unverified_initial_curve"}
    if pd.isna(anchor.exit_slot):
        return {**base, "status": "no_exit_anchor"}
    if anchor.entry_program != "pump" or anchor.exit_program != "pump":
        return {**base, "status": "pump_amm_migration_or_entry"}

    observed_state = CurveState()
    state = CurveState()
    our_tokens: int | None = None
    entry_net_lamports = _to_lamports(net_stake_sol)
    price_errors: list[float] = []
    try:
        for _, row in ordered.iterrows():
            position = _position(row)
            if position > exit_position:
                break
            if row.program != "pump":
                return {**base, "status": "pump_amm_migration_or_entry"}
            token_amount = _to_base_units(row.base_amount)
            if row.side == "buy":
                observed_state, observed_net_sol = observed_state.buy_exact_tokens(
                    token_amount
                )
            elif row.side == "sell":
                observed_state, observed_net_sol = observed_state.sell_exact_tokens(
                    token_amount
                )
            else:
                return {**base, "status": "unsupported_trade_side"}
            if row.price_sol > 0:
                price_error = (
                    abs(observed_state.price_sol - float(row.price_sol))
                    / float(row.price_sol)
                )
                price_errors.append(price_error)
                if price_error > 1e-6:
                    return {
                        **base,
                        "status": "unobserved_reserve_state_transition",
                        "observed_state_max_price_relative_error": max(price_errors),
                    }
            if our_tokens is None:
                state = observed_state
                if position == entry_position:
                    state, our_tokens = state.buy_exact_net_sol(entry_net_lamports)
                continue
            state = _apply_counterfactual_market_trade(
                state, row, buy_intent, observed_net_sol
            )
            if position == exit_position:
                state, gross_exit_lamports = state.sell_exact_tokens(our_tokens)
                entry_gross_sol, exit_net_sol, pnl_sol = strategy_cashflow(
                    net_stake_sol,
                    gross_exit_lamports / LAMPORTS_PER_SOL,
                    network_fee_sol,
                    fee_rate,
                )
                return {
                    **base,
                    "entry_net_sol": net_stake_sol,
                    "entry_gross_sol": entry_gross_sol,
                    "our_token_amount": our_tokens / BASE_UNITS_PER_TOKEN,
                    "exit_gross_sol": gross_exit_lamports / LAMPORTS_PER_SOL,
                    "exit_net_sol": exit_net_sol,
                    "network_fee_sol": network_fee_sol,
                    "pnl_sol": pnl_sol,
                    "net_roi": pnl_sol / net_stake_sol,
                    "observed_state_max_price_relative_error": max(price_errors, default=0.0),
                }
        return {**base, "status": "anchor_not_found_in_curve_events"}
    except ValueError as error:
        return {**base, "status": "counterfactual_curve_completed_or_invalid", "detail": str(error)}


def _build_anchors(predictions: pd.DataFrame) -> pd.DataFrame:
    carry = [
        column
        for column in (
            "label",
            "baseline_selected",
            "quality_selected",
            "two_stage_selected",
            "selective_two_stage_selected",
        )
        if column in predictions
    ]
    wanted_mask = np.zeros(len(predictions), dtype=bool)
    for column in carry:
        wanted_mask |= predictions[column].eq(1).to_numpy()
    wanted = predictions.loc[wanted_mask, ["token_address", *carry]].copy()
    con = _connection("36GB")
    con.register("wanted", wanted)
    carry_sql = ",".join(f"w.{column}" for column in carry)
    if carry_sql:
        carry_sql += ","
    return con.execute(
        f"""
        WITH trades AS MATERIALIZED (
          SELECT t.* FROM read_parquet('{_sql(JUNE_TRADES)}') t
          SEMI JOIN wanted USING(token_address)
          WHERE t.price_sol>0
        ), policies(latency_policy,tx_offset) AS (
          VALUES ('immediate',1),('offset_118',118)
        ), entry_candidates AS (
          SELECT w.token_address,{carry_sql}p.latency_policy,
                 t.block_slot entry_slot,t.tx_index entry_tx_index,
                 t.event_index entry_event_index,t.block_time entry_time,
                 t.program entry_program,
                 row_number() OVER(PARTITION BY w.token_address,p.latency_policy
                   ORDER BY t.block_slot,t.tx_index,t.event_index) rn
          FROM wanted w CROSS JOIN policies p JOIN trades t USING(token_address)
          WHERE (p.latency_policy='immediate' AND (
                   t.block_slot>t.deploy_block_slot OR
                   (t.block_slot=t.deploy_block_slot AND t.tx_index>t.deploy_tx_index)))
             OR (p.latency_policy='offset_118' AND t.block_slot=t.deploy_block_slot
                   AND t.tx_index>=t.deploy_tx_index+p.tx_offset)
        ), entries AS (SELECT * EXCLUDE(rn) FROM entry_candidates WHERE rn=1),
        exit_candidates AS (
          SELECT e.*,t.block_slot exit_slot,t.tx_index exit_tx_index,
                 t.event_index exit_event_index,t.block_time exit_time,
                 t.program exit_program,
                 row_number() OVER(PARTITION BY e.token_address,e.latency_policy
                   ORDER BY t.block_slot,t.tx_index,t.event_index) rn
          FROM entries e JOIN trades t USING(token_address)
          WHERE t.block_time>=e.entry_time+6
        )
        SELECT * EXCLUDE(rn) FROM exit_candidates WHERE rn=1
        """
    ).fetch_df()


def _load_curve_trades(anchors: pd.DataFrame) -> pd.DataFrame:
    maxima = (
        anchors.groupby("token_address", as_index=False)
        .agg(max_exit_slot=("exit_slot", "max"))
    )
    con = _connection("36GB")
    con.register("maxima", maxima)
    return con.execute(
        f"""
        SELECT t.token_address,t.block_slot,t.tx_index,t.event_index,t.block_time,
               t.side,t.program,t.price_sol,t.amount_sol,t.base_amount,
               t.deploy_block_slot,t.deploy_tx_index
        FROM read_parquet('{_sql(JUNE_TRADES)}') t
        JOIN maxima m USING(token_address)
        WHERE t.block_slot>=t.deploy_block_slot AND t.block_slot<=m.max_exit_slot
        ORDER BY t.token_address,t.block_slot,t.tx_index,t.event_index
        """
    ).fetch_df()


def _outcome_metrics(frame: pd.DataFrame, expected: int) -> dict[str, float | int]:
    ok = frame[frame.status.eq("ok")].sort_values(["entry_time", "token_address"])
    if ok.empty:
        return {"selected_tokens": expected, "supported_tokens": 0, "coverage": 0.0}
    pnl = ok.pnl_sol.to_numpy(dtype=float)
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = peak[1:] - equity
    capped = np.minimum(pnl, np.quantile(pnl, 0.99))
    return {
        "selected_tokens": expected,
        "supported_tokens": int(len(ok)),
        "coverage": float(len(ok) / expected) if expected else 0.0,
        "hit_rate": float((pnl > 0).mean()),
        "median_net_roi": float(ok.net_roi.median()),
        "mean_net_roi": float(ok.net_roi.mean()),
        "total_pnl_sol_supported": float(pnl.sum()),
        "p99_capped_total_pnl_sol_supported": float(capped.sum()),
        "max_drawdown_sol_supported": float(drawdown.max(initial=0.0)),
    }


def run_curve_replay(force: bool = False) -> dict[str, object]:
    """Replay supported standard Pump curves and report explicit bounds."""
    ensure_output_dirs()
    if not STRATEGY_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing {STRATEGY_PREDICTIONS}; run the profitable-disagreement experiment first"
        )
    predictions = pd.read_parquet(STRATEGY_PREDICTIONS)
    anchors = _build_anchors(predictions)
    trades = _load_curve_trades(anchors)
    stake_sol, _, network_fee_sol = _strategy_parameters()

    if CURVE_OUTCOMES.exists() and not force:
        outcomes = pd.read_parquet(CURVE_OUTCOMES)
    else:
        trade_groups = {token: frame for token, frame in trades.groupby("token_address", sort=False)}
        rows: list[dict[str, object]] = []
        for _, anchor in anchors.iterrows():
            token_trades = trade_groups.get(anchor.token_address)
            if token_trades is None:
                continue
            for buy_intent in ("fixed_quote", "fixed_token"):
                for fee_rate in (FEE_RATE_LOWER, FEE_RATE_UPPER):
                    row = replay_one(
                        token_trades,
                        anchor,
                        stake_sol,
                        network_fee_sol,
                        buy_intent,
                        fee_rate,
                    )
                    for column in (
                        "entry_time",
                        "label",
                        "baseline_selected",
                        "quality_selected",
                        "two_stage_selected",
                        "selective_two_stage_selected",
                    ):
                        if column in anchor:
                            row[column] = anchor[column]
                    rows.append(row)
        outcomes = pd.DataFrame(rows)
        outcomes.to_parquet(CURVE_OUTCOMES, index=False)

    coverage = (
        outcomes.groupby(["latency_policy", "buy_intent_bound", "fee_rate", "status"])
        .size()
        .rename("tokens")
        .reset_index()
    )
    coverage.to_csv(CURVE_COVERAGE, index=False)

    result: dict[str, object] = {
        "scope": "Integer replay for standard SOL-paired Pump bonding curves only; strategy orders are inserted after the entry and exit anchor events.",
        "official_mechanics": {
            "initial_virtual_token_reserves_raw": INITIAL_VIRTUAL_TOKEN_RESERVES,
            "initial_virtual_sol_reserves_lamports": INITIAL_VIRTUAL_SOL_RESERVES,
            "initial_real_token_reserves_raw": INITIAL_REAL_TOKEN_RESERVES,
            "formula": "Pump constant-product integer quotes; buy uses net SOL after swap fees.",
        },
        "bounds": {
            "observed_fee_rate_range": [FEE_RATE_LOWER, FEE_RATE_UPPER],
            "subsequent_buy_intent": ["fixed_quote", "fixed_token"],
            "sell_intent": "fixed token amount, as specified by the Pump sell instruction",
        },
        "frozen_strategy": {
            "net_curve_stake_sol": stake_sol,
            "round_trip_network_fee_sol": network_fee_sol,
            "hold_seconds": 6,
        },
        "results": {},
        "limitations": [
            "The normalized trade table omits TradeEvent ix_name, reserve, fee, Mayhem, and cashback fields.",
            "Nonstandard/Mayhem initial curves and any PumpSwap migration before exit are excluded rather than guessed.",
            "Counterfactual market buys are replayed under both fixed-quote and fixed-token intent; raw instructions are needed to identify the realized branch.",
            "A counterfactual that would complete the curve is excluded because the supplied table does not support exact migration replay.",
            "Slippage-limit failures caused by our inserted order cannot be identified without raw instructions.",
        ],
    }
    policies = {
        "baseline_replica": "baseline_selected",
        "quality_augmented_replica": "quality_selected",
        "two_stage": "two_stage_selected",
        "selective_two_stage": "selective_two_stage_selected",
        "target_equal_stake": "label",
    }
    for latency in ("immediate", "offset_118"):
        latency_result: dict[str, object] = {}
        for cohort, column in policies.items():
            expected = int(predictions[column].sum())
            cohort_result: dict[str, object] = {}
            for buy_intent in ("fixed_quote", "fixed_token"):
                intent_result: dict[str, object] = {}
                for fee_rate in (FEE_RATE_LOWER, FEE_RATE_UPPER):
                    subset = outcomes[
                        outcomes.latency_policy.eq(latency)
                        & outcomes.buy_intent_bound.eq(buy_intent)
                        & outcomes.fee_rate.eq(fee_rate)
                        & outcomes[column].eq(1)
                    ]
                    intent_result[f"fee_{fee_rate:.4f}"] = _outcome_metrics(subset, expected)
                cohort_result[buy_intent] = intent_result
            latency_result[cohort] = cohort_result
        result["results"][latency] = latency_result  # type: ignore[index]
    CURVE_RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
