from __future__ import annotations

import numpy as np
import pandas as pd


# The targeted raw-block sample establishes a 95 bp Pump protocol component and
# a separate 30 bp target-route transfer. The latter transfer's semantic label is
# not established by the supplied data, so the ledger deliberately calls it a
# raw-observed fallback rather than a creator, routing, or referral fee.
PUMP_TARGET_ROUTE_TOTAL_RATE = 0.0125

NUMERIC_FIELDS = (
    "cost_usd",
    "quote_amount",
    "gas_usd",
    "gas_native",
    "dex_usd",
    "dex_native",
    "priority_fee",
    "tip_fee",
)


def _first_nonempty(series: pd.Series) -> str:
    values = series.fillna("").astype(str)
    values = values[values.ne("")]
    return values.iloc[0] if len(values) else ""


def build_transaction_fee_ledger(activity: pd.DataFrame) -> pd.DataFrame:
    """Build one fee row per transaction from the target activity table.

    ``cost_usd`` remains quote-principal cash flow. ``gas_*`` is charged once
    per transaction and already contains ``priority_fee`` and ``tip_fee``.
    Pump trades receive the raw-supported 1.25% target-route cost separately.
    Reported DEX fields for other routed venues are retained as diagnostics but
    are not charged again because the observed wallet quote transfer is already
    the net cash flow. Blank-venue DEX fields remain explicitly ambiguous.
    """
    required = {
        "tx_hash",
        "token_address",
        "event_type",
        "launchpad",
        "quote_token_symbol",
        *NUMERIC_FIELDS,
    }
    missing = sorted(required.difference(activity.columns))
    if missing:
        raise ValueError(f"fee ledger missing required columns: {missing}")

    frame = activity.copy()
    for field in NUMERIC_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce").fillna(0.0)
    for field in ("tx_hash", "token_address", "event_type", "launchpad", "quote_token_symbol"):
        frame[field] = frame[field].fillna("").astype(str)

    token_counts = frame.groupby("tx_hash").token_address.apply(
        lambda values: values[values.ne("")].nunique()
    )
    if token_counts.gt(1).any():
        examples = token_counts[token_counts.gt(1)].index[:3].tolist()
        raise ValueError(
            "cannot allocate a transaction fee across multiple observed tokens: "
            f"{examples}"
        )

    trade = frame[frame.event_type.isin(["buy", "sell"])].copy()
    trade_counts = trade.groupby("tx_hash").size()
    if trade_counts.gt(1).any():
        examples = trade_counts[trade_counts.gt(1)].index[:3].tolist()
        raise ValueError(
            "multiple trade rows in one transaction need an explicit allocation rule: "
            f"{examples}"
        )

    transaction = frame.groupby("tx_hash", as_index=False).agg(
        token_address=("token_address", _first_nonempty),
        network_cost_usd=("gas_usd", "max"),
        network_cost_native=("gas_native", "max"),
        priority_fee_included_native=("priority_fee", "max"),
        tip_fee_included_native=("tip_fee", "max"),
        reported_dex_usd=("dex_usd", "max"),
        reported_dex_native=("dex_native", "max"),
    )
    transaction["network_base_and_other_included_native"] = (
        transaction.network_cost_native
        - transaction.priority_fee_included_native
        - transaction.tip_fee_included_native
    ).clip(lower=0.0)

    trade_fields = trade[
        [
            "tx_hash",
            "event_type",
            "launchpad",
            "quote_token_symbol",
            "cost_usd",
            "quote_amount",
        ]
    ].rename(columns={"cost_usd": "quote_principal_usd"})
    transaction = transaction.merge(trade_fields, on="tx_hash", how="left", validate="one_to_one")
    transaction["launchpad"] = transaction.launchpad.fillna("")
    transaction["event_type"] = transaction.event_type.fillna("")
    transaction["quote_token_symbol"] = transaction.quote_token_symbol.fillna("")
    transaction[["quote_principal_usd", "quote_amount"]] = transaction[
        ["quote_principal_usd", "quote_amount"]
    ].fillna(0.0)

    is_pump_trade = (
        transaction.event_type.isin(["buy", "sell"])
        & transaction.launchpad.str.lower().eq("pump")
    )
    is_native_quote = transaction.quote_token_symbol.str.upper().isin(["SOL", "WSOL"])
    known_routed_trade = (
        transaction.event_type.isin(["buy", "sell"])
        & transaction.launchpad.ne("")
        & ~is_pump_trade
    )
    unidentified_trade = transaction.event_type.isin(["buy", "sell"]) & transaction.launchpad.eq("")

    transaction["pump_separate_cost_usd"] = np.where(
        is_pump_trade,
        transaction.quote_principal_usd * PUMP_TARGET_ROUTE_TOTAL_RATE,
        0.0,
    )
    transaction["pump_separate_cost_native"] = np.where(
        is_pump_trade & is_native_quote,
        transaction.quote_amount * PUMP_TARGET_ROUTE_TOTAL_RATE,
        0.0,
    )
    transaction["pump_reported_component_usd"] = np.where(
        is_pump_trade,
        np.minimum(transaction.reported_dex_usd, transaction.pump_separate_cost_usd),
        0.0,
    )
    transaction["pump_reported_component_native"] = np.where(
        is_pump_trade,
        np.minimum(
            transaction.reported_dex_native,
            transaction.pump_separate_cost_native,
        ),
        0.0,
    )
    transaction["pump_raw_observed_fallback_usd"] = (
        transaction.pump_separate_cost_usd - transaction.pump_reported_component_usd
    )
    transaction["pump_raw_observed_fallback_native"] = (
        transaction.pump_separate_cost_native - transaction.pump_reported_component_native
    )
    transaction["pump_reported_residual_ambiguous_usd"] = np.where(
        is_pump_trade,
        (transaction.reported_dex_usd - transaction.pump_reported_component_usd).clip(lower=0.0),
        0.0,
    )
    transaction["routed_dex_contained_in_quote_usd"] = np.where(
        known_routed_trade, transaction.reported_dex_usd, 0.0
    )
    transaction["routed_dex_contained_in_quote_native"] = np.where(
        known_routed_trade, transaction.reported_dex_native, 0.0
    )
    transaction["unidentified_dex_ambiguous_usd"] = np.where(
        unidentified_trade, transaction.reported_dex_usd, 0.0
    )
    transaction["unidentified_dex_ambiguous_native"] = np.where(
        unidentified_trade, transaction.reported_dex_native, 0.0
    )
    transaction["total_defensible_cost_usd"] = (
        transaction.network_cost_usd + transaction.pump_separate_cost_usd
    )
    transaction["total_defensible_cost_native"] = (
        transaction.network_cost_native + transaction.pump_separate_cost_native
    )
    return transaction.sort_values("tx_hash").reset_index(drop=True)
