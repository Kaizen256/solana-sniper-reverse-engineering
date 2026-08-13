from __future__ import annotations

import pytest

from solana_sniper_reverse_engineering.curve_replay import (
    BASE_UNITS_PER_TOKEN,
    INITIAL_REAL_SOL_RESERVES,
    INITIAL_REAL_TOKEN_RESERVES,
    INITIAL_VIRTUAL_SOL_RESERVES,
    INITIAL_VIRTUAL_TOKEN_RESERVES,
    LAMPORTS_PER_SOL,
    CurveState,
    strategy_cashflow,
)


def test_buy_reserve_accounting_and_price_direction() -> None:
    before = CurveState()
    after, tokens = before.buy_exact_net_sol(1_975_308_641)
    assert after.virtual_sol_reserves == INITIAL_VIRTUAL_SOL_RESERVES + 1_975_308_641
    assert after.real_sol_reserves == INITIAL_REAL_SOL_RESERVES + 1_975_308_641
    assert after.virtual_token_reserves == INITIAL_VIRTUAL_TOKEN_RESERVES - tokens
    assert after.real_token_reserves == INITIAL_REAL_TOKEN_RESERVES - tokens
    assert after.price_sol > before.price_sol
    assert (
        after.virtual_sol_reserves * after.virtual_token_reserves
        >= before.virtual_sol_reserves * before.virtual_token_reserves
    )


def test_sell_reverses_reserves_and_lowers_price() -> None:
    initial = CurveState()
    bought, tokens = initial.buy_exact_net_sol(1_975_308_641)
    sold, gross_sol = bought.sell_exact_tokens(tokens)
    assert sold.price_sol < bought.price_sol
    assert sold.virtual_token_reserves == initial.virtual_token_reserves
    assert sold.real_token_reserves == initial.real_token_reserves
    # Pump's documented -1/+1 integer guards leave one lamport on the curve.
    assert gross_sol == 1_975_308_640
    assert sold.virtual_sol_reserves == initial.virtual_sol_reserves + 1
    assert sold.real_sol_reserves == 1


def test_larger_buy_has_monotonically_worse_average_execution() -> None:
    state = CurveState()
    _, small_tokens = state.buy_exact_net_sol(LAMPORTS_PER_SOL)
    _, large_tokens = state.buy_exact_net_sol(2 * LAMPORTS_PER_SOL)
    assert large_tokens > small_tokens
    assert large_tokens / 2 < small_tokens


def test_buy_reverse_quote_round_trip() -> None:
    state = CurveState()
    after, tokens = state.buy_exact_net_sol(1_975_308_641)
    reverse_after, inferred_sol = state.buy_exact_tokens(tokens)
    assert inferred_sol == 1_975_308_641
    assert reverse_after == after


def test_fee_accounting_is_symmetric_and_monotone() -> None:
    low = strategy_cashflow(1.975308641, 2.4, 0.09101, 0.0095)
    high = strategy_cashflow(1.975308641, 2.4, 0.09101, 0.0125)
    assert low[0] < high[0]
    assert low[1] > high[1]
    assert low[2] > high[2]
    assert low[2] == pytest.approx(low[1] - low[0] - 0.09101)


def test_known_normalized_standard_curve_event() -> None:
    # Observed June deployment event: a 50M-token developer buy on the standard
    # curve. This validates integer units and the emitted post-trade price.
    state = CurveState()
    after, net_sol = state.buy_exact_tokens(50_000_000 * BASE_UNITS_PER_TOKEN)
    # The normalized doubles round the emitted integer token amount, leaving a
    # one-lamport reverse-quote interval.
    assert abs(net_sol - 1_466_275_660) <= 1
    assert after.price_sol == pytest.approx(3.07588227370479e-8, rel=1e-10)


def test_invalid_curve_operations_are_rejected() -> None:
    state = CurveState()
    with pytest.raises(ValueError):
        state.buy_exact_net_sol(1)
    with pytest.raises(ValueError):
        state.buy_exact_tokens(INITIAL_REAL_TOKEN_RESERVES + 1)
    with pytest.raises(ValueError):
        state.sell_exact_tokens(BASE_UNITS_PER_TOKEN)
