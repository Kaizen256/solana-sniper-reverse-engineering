from __future__ import annotations

import pandas as pd
import pytest

from solana_sniper_reverse_engineering.fee_ledger import (
    PUMP_TARGET_ROUTE_TOTAL_RATE,
    build_transaction_fee_ledger,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "tx_hash": "tx",
        "token_address": "token",
        "event_type": "buy",
        "launchpad": "pump",
        "quote_token_symbol": "SOL",
        "cost_usd": 100.0,
        "quote_amount": 2.0,
        "gas_usd": 2.0,
        "gas_native": 0.01,
        "dex_usd": 0.95,
        "dex_native": 0.019,
        "priority_fee": 0.004,
        "tip_fee": 0.001,
    }
    row.update(overrides)
    return row


def test_pump_ledger_charges_principal_network_and_route_total_once() -> None:
    ledger = build_transaction_fee_ledger(pd.DataFrame([_row()])).iloc[0]
    assert ledger.quote_principal_usd == pytest.approx(100.0)
    assert ledger.network_cost_usd == pytest.approx(2.0)
    assert ledger.priority_fee_included_native == pytest.approx(0.004)
    assert ledger.tip_fee_included_native == pytest.approx(0.001)
    assert ledger.pump_separate_cost_usd == pytest.approx(
        100.0 * PUMP_TARGET_ROUTE_TOTAL_RATE
    )
    assert ledger.pump_reported_component_usd == pytest.approx(0.95)
    assert ledger.pump_raw_observed_fallback_usd == pytest.approx(0.30)
    # Priority/tip are components of gas, not additional charges.
    assert ledger.total_defensible_cost_usd == pytest.approx(3.25)


def test_pump_reported_total_is_not_added_on_top_of_route_total() -> None:
    ledger = build_transaction_fee_ledger(
        pd.DataFrame([_row(dex_usd=1.25, dex_native=0.025)])
    ).iloc[0]
    assert ledger.pump_reported_component_usd == pytest.approx(1.25)
    assert ledger.pump_raw_observed_fallback_usd == pytest.approx(0.0)
    assert ledger.total_defensible_cost_usd == pytest.approx(3.25)


def test_routed_dex_is_retained_but_not_double_subtracted() -> None:
    ledger = build_transaction_fee_ledger(
        pd.DataFrame([_row(launchpad="ray_launchpad", dex_usd=3.0, dex_native=0.03)])
    ).iloc[0]
    assert ledger.pump_separate_cost_usd == 0
    assert ledger.routed_dex_contained_in_quote_usd == pytest.approx(3.0)
    assert ledger.total_defensible_cost_usd == pytest.approx(2.0)


def test_unidentified_dex_remains_ambiguous_and_uncharged() -> None:
    ledger = build_transaction_fee_ledger(
        pd.DataFrame([_row(launchpad="", dex_usd=0.4, dex_native=0.004)])
    ).iloc[0]
    assert ledger.unidentified_dex_ambiguous_usd == pytest.approx(0.4)
    assert ledger.total_defensible_cost_usd == pytest.approx(2.0)


def test_non_trade_row_does_not_duplicate_transaction_network_cost() -> None:
    rows = [
        _row(),
        _row(
            event_type="launch",
            cost_usd=0,
            quote_amount=0,
            dex_usd=0,
            dex_native=0,
        ),
    ]
    ledger = build_transaction_fee_ledger(pd.DataFrame(rows))
    assert len(ledger) == 1
    assert ledger.iloc[0].network_cost_usd == pytest.approx(2.0)
    assert ledger.iloc[0].pump_separate_cost_usd == pytest.approx(1.25)


def test_multi_token_transaction_requires_explicit_allocation() -> None:
    rows = [_row(), _row(event_type="burn", token_address="another")]
    with pytest.raises(ValueError, match="multiple observed tokens"):
        build_transaction_fee_ledger(pd.DataFrame(rows))
