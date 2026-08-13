from __future__ import annotations

import json
from pathlib import Path
import re

import duckdb
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "deployment_features.parquet"
OUTCOMES = ROOT / "artifacts" / "tables" / "june_strategy_outcomes.parquet"
INTRASLOT = ROOT / "artifacts" / "tables" / "june_intraslot_outcomes.parquet"
QUALITY = ROOT / "data" / "interim" / "historical_launch_quality_features.parquet"
DEV_SELL = ROOT / "data" / "interim" / "historical_launch_dev_sell_features.parquet"
FINAL_IMPORTANCE = ROOT / "submission" / "tables" / "final_selector_feature_importance.csv"
FINAL_EFFECTS = ROOT / "submission" / "tables" / "final_selector_feature_effects.csv"
HEAD_TO_HEAD = ROOT / "submission" / "tables" / "third_pass_head_to_head.csv"
CURVE_RESULTS = ROOT / "submission" / "tables" / "curve_replay_results.json"
NOTEBOOK = ROOT / "submission" / "final_notebook.ipynb"


def test_final_presentation_tables_use_promoted_selector_and_include_target() -> None:
    importance = pd.read_csv(FINAL_IMPORTANCE)
    effects = pd.read_csv(FINAL_EFFECTS)
    head_to_head = pd.read_csv(HEAD_TO_HEAD)
    expected_top10 = [
        "seconds_since_prior_deploy",
        "dev_buy_sol",
        "hist_quote_sol_sum",
        "hist_open_close_count",
        "hist_cost_usd_sum",
        "latest_prior_launch_dev_sell_latency_seconds",
        "prior_deploy_count_1d",
        "hist_burn_count",
        "hist_burn_count_30d",
        "hist_claim_fee_usd_per_claimed_launch",
    ]
    assert importance.head(10).feature.tolist() == expected_top10
    assert set(effects.feature) == set(expected_top10)
    assert "target_equal_stake_counterfactual" in set(head_to_head.policy)
    for column in (
        "marginal_immediate_hit_rate",
        "marginal_offset118_hit_rate",
        "marginal_offset118_median_roi",
        "marginal_offset118_p99_capped_pnl_sol",
        "marginal_offset118_max_drawdown_sol",
    ):
        assert head_to_head[column].notna().all()


def test_executed_notebook_has_clean_native_figures_and_separate_execution_tables() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    bootstrap = sources[2]
    assert bootstrap.count("from pathlib import Path") == 1
    assert bootstrap.count("import json") == 1
    assert "cwd / 'pyproject.toml'" in bootstrap
    assert "cwd.parent / 'pyproject.toml'" in bootstrap
    assert "from IPython.display import Image, display" in bootstrap
    assert "plt.show" not in bootstrap

    expected_figures = {
        "02_entry_latency.png",
        "03_holds_and_exits.png",
        "05_model_diagnostics.png",
        "06_backtest_comparison.png",
        "07_third_pass_summary.png",
    }
    source_blob = "\n".join(sources)
    assert all(figure in source_blob for figure in expected_figures)

    outputs = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    ]
    embedded_pngs = [
        output for output in outputs if "image/png" in output.get("data", {})
    ]
    assert len(embedded_pngs) == len(expected_figures)
    output_blob = json.dumps(outputs)
    assert "FigureCanvasAgg" not in output_blob
    assert not re.search(r"/(?:home|Users)/[^/]+/", output_blob)
    assert not re.search(r"[A-Za-z]:\\\\Users\\\\[^\\\\]+", output_blob)

    marginal_heading = sources.index(next(s for s in sources if s.startswith("## 8. Marginal +118")))
    exact_heading = sources.index(next(s for s in sources if s.startswith("## 9. Exact Pump")))
    assert "not exact curve replay" in sources[marginal_heading].lower()
    assert "third_pass_head_to_head.csv" in sources[marginal_heading + 1]
    assert "run_curve_replay" not in sources[marginal_heading + 1]
    assert "run_curve_replay" in sources[exact_heading + 1]
    assert "curve_replay_results.json" in sources[exact_heading + 1]
    assert "curve_replay_coverage.csv" in sources[exact_heading + 1]
    assert "50.58" not in sources[exact_heading + 1]

    exact_outputs = notebook["cells"][exact_heading + 1].get("outputs", [])
    exact_output_blob = json.dumps(exact_outputs)
    curve = json.loads(CURVE_RESULTS.read_text())
    selective = curve["results"]["offset_118"]["selective_two_stage"]
    for intent in ("fixed_quote", "fixed_token"):
        for fee in ("fee_0.0095", "fee_0.0125"):
            metrics = selective[intent][fee]
            assert f"{metrics['coverage']:.2%}" in exact_output_blob
            assert f"{metrics['median_net_roi']:.2%}" in exact_output_blob
            assert f"{metrics['p99_capped_total_pnl_sol_supported']:+.1f}" in exact_output_blob


@pytest.mark.skipif(not FEATURES.exists(), reason="generated full feature store not present")
def test_full_feature_store_cardinality_and_temporal_gate() -> None:
    con = duckdb.connect()
    result = con.execute(
        """
        SELECT count(*) AS n_rows, count(DISTINCT token_address) AS tokens,
               count(*)-count(DISTINCT (tx_hash, token_address)) duplicate_keys,
               sum(label) positives,
               count(*) FILTER (WHERE seconds_since_activity < 1) invalid_activity,
               count(*) FILTER (WHERE seconds_since_prior_deploy < 1) invalid_deploy
        FROM read_parquet(?)
        """,
        [str(FEATURES)],
    ).fetchone()
    assert result == (5_076_421, 5_076_421, 0, 15_927, 0, 0)


@pytest.mark.skipif(not OUTCOMES.exists(), reason="generated June backtest outcomes not present")
def test_backtest_outcome_ordering_and_keys() -> None:
    con = duckdb.connect()
    result = con.execute(
        """
        SELECT count(*)-count(DISTINCT (token_address, delay_slots)) duplicate_keys,
               count(*) FILTER (
                 WHERE exit_time IS NOT NULL AND exit_time < entry_time + 6
               ) invalid_exits,
               count(*) FILTER (WHERE entry_price_sol <= 0) invalid_entry_prices,
               count(*) FILTER (
                 WHERE exit_price_sol IS NOT NULL AND exit_price_sol <= 0
               ) invalid_exit_prices
        FROM read_parquet(?)
        """,
        [str(OUTCOMES)],
    ).fetchone()
    assert result == (0, 0, 0, 0)


@pytest.mark.skipif(not INTRASLOT.exists(), reason="generated intra-slot outcomes not present")
def test_intraslot_outcomes_respect_transaction_offset() -> None:
    con = duckdb.connect()
    result = con.execute(
        """
        SELECT count(*)-count(DISTINCT (token_address, latency_policy)) duplicate_keys,
               count(*) FILTER (WHERE entry_slot <> deploy_block_slot) wrong_slot,
               count(*) FILTER (
                 WHERE entry_tx_index < deploy_tx_index + tx_offset
               ) early_entries,
               count(*) FILTER (
                 WHERE exit_time IS NOT NULL AND exit_time < entry_time + 6
               ) early_exits
        FROM read_parquet(?)
        """,
        [str(INTRASLOT)],
    ).fetchone()
    assert result == (0, 0, 0, 0)


@pytest.mark.skipif(not QUALITY.exists(), reason="generated historical quality features not present")
def test_historical_quality_features_are_strict_point_in_time() -> None:
    con = duckdb.connect()
    result = con.execute(
        """
        SELECT count(*) AS rows,count(DISTINCT token_address) tokens,
               count(*) FILTER (WHERE quality_state_time>=f.block_time) future_state,
               count(*) FILTER (WHERE latest_prior_launch_time>=f.block_time) future_launch,
               count(*) FILTER (WHERE seconds_since_claim_fee<1) same_second_claim,
               count(*) FILTER (
                 WHERE hist_mature_1d_success_count>hist_mature_1d_launch_count
                    OR hist_mature_7d_success_count>hist_mature_7d_launch_count
               ) impossible_success_counts,
               count(*) FILTER (
                 WHERE hist_claimed_launch_fraction NOT BETWEEN 0 AND 1
                    OR hist_claimed_launch_per_core_deploy NOT BETWEEN 0 AND 1
                    OR hist_decayed_30d_success_fraction NOT BETWEEN 0 AND 1
               ) invalid_bounded_fractions,
               count(*) FILTER (
                 WHERE quality_launch_history_incomplete <>
                   (hist_claimed_launch_count>hist_pump_launch_count)::UTINYINT
               ) invalid_incomplete_flags
        FROM read_parquet(?) q JOIN read_parquet(?) f USING(token_address)
        """,
        [str(QUALITY), str(FEATURES)],
    ).fetchone()
    assert result == (5_076_421, 5_076_421, 0, 0, 0, 0, 0, 0)


@pytest.mark.skipif(not DEV_SELL.exists(), reason="generated developer-sell features not present")
def test_developer_sell_features_are_strict_point_in_time() -> None:
    con = duckdb.connect()
    result = con.execute(
        """
        SELECT count(*) AS rows,count(DISTINCT d.token_address) AS tokens,
               count(*) FILTER (WHERE dev_sell_state_time>=f.block_time) AS future_state,
               count(*) FILTER (WHERE latest_dev_sell_launch_time>=f.block_time) AS future_launch,
               count(*) FILTER (WHERE seconds_since_prior_dev_sell<1) AS same_second_sell,
               count(*) FILTER (
                 WHERE hist_mature_1d_dev_sold_count>q.hist_mature_1d_launch_count
                    OR hist_mature_7d_dev_sold_count>q.hist_mature_7d_launch_count
                    OR hist_dev_sold_launch_fraction NOT BETWEEN 0 AND 1
                    OR hist_mature_1d_dev_sold_fraction NOT BETWEEN 0 AND 1
                    OR hist_mature_7d_dev_sold_fraction NOT BETWEEN 0 AND 1
               ) AS invalid_fractions
        FROM read_parquet(?) d
        JOIN read_parquet(?) f USING(token_address)
        JOIN read_parquet(?) q USING(token_address)
        """,
        [str(DEV_SELL), str(FEATURES), str(QUALITY)],
    ).fetchone()
    assert result == (5_076_421, 5_076_421, 0, 0, 0, 0)
