from __future__ import annotations

import duckdb
import json
from pathlib import Path

import pytest

from solana_sniper_reverse_engineering.target_relationship import (
    ALL_RELATIONSHIP_FEATURES,
    CORE_RELATIONSHIP_FEATURES,
)
from solana_sniper_reverse_engineering.third_pass import DEVELOPER_SELL_FEATURES


ROOT = Path(__file__).resolve().parents[1]
FULL_FEATURES = ROOT / "data/interim/target_signer_relationship_features_full.parquet"
JUNE_REPORT = ROOT / "submission/tables/target_relationship_june_reporting.json"
SOURCE_ROLES = ROOT / "submission/tables/target_relationship_source_roles.json"
REPRODUCTION_RECIPE = ROOT / "submission/tables/target_relationship_reproduction_recipe.json"


def test_relationship_allowlist_contains_only_historical_state() -> None:
    assert set(CORE_RELATIONSHIP_FEATURES) <= set(ALL_RELATIONSHIP_FEATURES)
    prohibited = {"label", "token_address", "tx_hash", "tx_signer", "current_target_buy"}
    assert not (set(ALL_RELATIONSHIP_FEATURES) & prohibited)
    assert all(
        name.startswith(("prior_", "seconds_since_", "deployments_since_"))
        or name == "target_signer_known"
        for name in ALL_RELATIONSHIP_FEATURES
    )


def test_target_state_excludes_current_and_future_same_second_events() -> None:
    con = duckdb.connect()
    rows = con.execute(
        """
        WITH candidates(token,t) AS (VALUES ('early',100),('later',101)),
        target_events(t) AS (VALUES (90),(100),(110)),
        state AS (
          SELECT t,count(*) OVER (
            ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          ) cumulative_buys
          FROM target_events
        )
        SELECT c.token,coalesce(s.cumulative_buys,0) prior_buys,c.t-s.t recency
        FROM candidates c ASOF LEFT JOIN state s ON c.t>s.t
        ORDER BY c.t
        """
    ).fetchall()
    assert rows == [("early", 1, 10), ("later", 2, 1)]


def test_target_rolling_count_includes_left_boundary_and_excludes_decision_second() -> None:
    con = duckdb.connect()
    value = con.execute(
        """
        WITH events(t,n) AS (VALUES (0,1),(1,1),(3600,1)),
        state AS (
          SELECT t,sum(n) OVER (
            ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          ) cumulative
          FROM events
        ), decision(t) AS (VALUES (3600))
        SELECT coalesce(current.cumulative,0)-coalesce(before_window.cumulative,0)
        FROM decision d
        ASOF LEFT JOIN state current ON d.t>current.t
        ASOF LEFT JOIN state before_window ON d.t-3600>before_window.t
        """
    ).fetchone()[0]
    assert value == 2


@pytest.mark.skipif(not FULL_FEATURES.exists(), reason="generated relationship store not present")
def test_generated_relationship_store_has_full_keys_and_strict_state() -> None:
    con = duckdb.connect()
    result = con.execute(
        """
        SELECT count(*) AS n_rows,count(DISTINCT r.token_address) AS tokens,
               count(*) FILTER (
                 WHERE r.target_relationship_state_time IS NOT NULL
                   AND r.target_relationship_state_time>=f.block_time
               ) future_or_equal,
               count(*) FILTER (WHERE r.seconds_since_prior_target_buy<=0) invalid_recency
        FROM read_parquet(?) r
        JOIN read_parquet(?) f USING(token_address)
        """,
        [str(FULL_FEATURES), str(ROOT / "data/processed/deployment_features.parquet")],
    ).fetchone()
    assert result == (5_076_421, 5_076_421, 0, 0)


@pytest.mark.skipif(not JUNE_REPORT.exists(), reason="frozen June report not present")
def test_june_report_is_frozen_and_not_a_selection_input() -> None:
    result = json.loads(JUNE_REPORT.read_text())
    assert result["status"] == "FROZEN_JUNE_REPORT"
    assert result["no_post_june_redesign"] is True
    assert result["metrics"]["rows"] == 852_083
    assert result["metrics"]["positives"] == 4_195
    assert result["metrics"]["pr_auc"] == pytest.approx(0.20471037711709195)


def test_rescue_source_roles_keep_current_and_future_target_reactions_out() -> None:
    roles = json.loads(SOURCE_ROLES.read_text())
    assert "strictly earlier target-wallet transactions used only as online state for later candidates" in roles["A_entry_features"]["sources"]
    assert "current candidate target-wallet reaction" in roles["B_labels_evaluation"]
    assert "future target-wallet activity relative to a candidate" in roles["D_behavior_only"]
    assert "same-second target or deployer activity" in roles["E_excluded_ambiguous"]


def test_public_reproduction_recipe_is_the_frozen_recipe() -> None:
    recipe = json.loads(REPRODUCTION_RECIPE.read_text())
    assert recipe["training_window"]["start_unix"] == 1_773_273_600
    assert recipe["training_window"]["end_unix"] == 1_777_593_600
    assert recipe["june_reporting_window"]["start_unix"] == 1_780_272_000
    assert recipe["june_reporting_window"]["end_unix"] == 1_782_864_000
    assert recipe["negative_sampling"] == {
        "procedure": "retain every positive and a negative iff DuckDB hash(token_address) % 2 = 0",
        "stride": 2,
        "positive_sample_weight": 1.0,
        "sampled_negative_weight": 2.0,
        "row_order": "block_time, token_address",
    }
    assert recipe["threshold"] == 0.23211809647507783
    assert recipe["features"]["selected_relationship"] == ALL_RELATIONSHIP_FEATURES
    assert recipe["expected"]["feature_state"]["sampled_training_rows"] == 650_194
    assert recipe["expected"]["june_metrics"]["predicted_entries"] == 6_094
    assert recipe["expected"]["june_metrics"]["true_positives"] == 1_787


def test_final_relationship_schema_honors_developer_sell_drop_gate() -> None:
    decision = json.loads(
        (ROOT / "submission/tables/developer_sell_outcome_results.json").read_text()
    )["pre_june_decision"]
    freeze = json.loads(
        (ROOT / "artifacts/models/target_relationship_rescue/freeze_manifest.json").read_text()
    )
    assert decision["status"] == "DROP"
    assert not (set(freeze["numeric_features"]) & set(DEVELOPER_SELL_FEATURES))
