from __future__ import annotations

import pandas as pd
import pytest

from solana_sniper_reverse_engineering.backtest import _metrics


def test_fixed_stake_pnl_and_drawdown_accounting() -> None:
    cohort = pd.DataFrame(
        {
            "entry_time": [1, 2, 3],
            "token_address": ["a", "b", "c"],
            "gross_roi": [1.0, -0.5, 0.25],
            "forced_total_loss": [0, 0, 0],
        }
    )
    metrics, equity = _metrics(
        cohort, expected_tokens=4, stake_sol=2.0, network_cost_sol=0.1
    )
    assert metrics["fill_rate"] == pytest.approx(0.75)
    assert list(equity.pnl_sol) == pytest.approx([1.9, -1.1, 0.4])
    assert "network_cost_adjusted_roi" in equity
    assert "net_roi" not in equity
    assert metrics["total_pnl_sol"] == pytest.approx(1.2)
    assert metrics["max_drawdown_sol"] == pytest.approx(1.1)
    assert metrics["total_pnl_sol_roi_capped_at_p99"] <= metrics["total_pnl_sol"]


def test_missing_exit_total_loss_is_counted() -> None:
    cohort = pd.DataFrame(
        {
            "entry_time": [1],
            "token_address": ["a"],
            "gross_roi": [-1.0],
            "forced_total_loss": [1],
        }
    )
    metrics, _ = _metrics(
        cohort, expected_tokens=1, stake_sol=1.0, network_cost_sol=0.0
    )
    assert metrics["forced_total_losses"] == 1
    assert metrics["total_pnl_sol"] == pytest.approx(-1.0)
