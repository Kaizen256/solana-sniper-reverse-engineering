#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from solana_sniper_reverse_engineering.config import (
    BOUGHT_TXS,
    INTERIM,
    PROCESSED,
    ROOT,
    SUBMISSION,
    TARGET_TX_INDEX,
)
from solana_sniper_reverse_engineering.behavior import TARGET_POSITIONS, TOKEN_METRICS
from solana_sniper_reverse_engineering.feature_store import (
    ACTIVITY_STATE,
    BASE_FEATURES,
    FEATURE_MANIFEST,
    FEATURE_STORE,
    audit as audit_features,
)
from solana_sniper_reverse_engineering.message_features import extract_file
from solana_sniper_reverse_engineering.frozen_reproduction import (
    EXACT_REPLAY_REPORT,
    RECIPE as REPRODUCTION_RECIPE,
    REPRODUCTION_FEATURE_MANIFEST,
    REPRODUCTION_FEATURES,
    REPRODUCTION_MODEL,
    REPRODUCTION_PREDICTIONS,
    REPRODUCTION_REPORT,
)


def _assert_manifest_source(item: dict[str, object]) -> None:
    source = Path(str(item["path"] if "path" in item else item["source"]))
    assert source.exists(), source
    assert source.stat().st_size == int(
        item["bytes"] if "bytes" in item else item["source_bytes"]
    ), source
    recorded_mtime = item.get("mtime_ns", item.get("source_mtime_ns"))
    assert source.stat().st_mtime_ns == int(recorded_mtime), source


def audit_manifests() -> dict[str, object]:
    message_rows = 0
    for name in ("bought", "not_bought"):
        output = INTERIM / "message_features" / f"{name}.parquet"
        manifest_path = output.with_suffix(".manifest.json")
        assert output.exists() and manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        _assert_manifest_source(manifest)
        rows = pq.ParquetFile(output).metadata.num_rows
        assert rows == int(manifest["token_row_count"])
        message_rows += rows

    manifest = json.loads(FEATURE_MANIFEST.read_text())
    for source in manifest["sources"]:
        _assert_manifest_source(source)
    assert pq.ParquetFile(FEATURE_STORE).metadata.num_rows == int(manifest["row_count"])
    assert BASE_FEATURES.exists() and ACTIVITY_STATE.exists()
    checks = audit_features(FEATURE_STORE)
    assert checks == {
        "rows": 5_076_421,
        "tokens": 5_076_421,
        "duplicate_keys": 0,
        "positives": 15_927,
        "missing_history": 1_442_833,
        "missing_message": 6_274,
        "invalid_activity_recency": 0,
        "invalid_deploy_recency": 0,
    }
    con = duckdb.connect()
    target_positions = con.execute(
        """
        SELECT count(*) positions, count(DISTINCT p.tx_hash) distinct_positions,
               count(*) FILTER (WHERE i.tx_hash IS NULL) unmatched,
               count(*) FILTER (
                 WHERE p.block_time <> i.blockTime OR p.block_slot <> i.blockSlot
               ) time_slot_mismatches
        FROM read_parquet(?) p
        LEFT JOIN read_parquet(?) i USING (tx_hash)
        """,
        [str(TARGET_POSITIONS), str(TARGET_TX_INDEX)],
    ).fetchone()
    assert target_positions == (87_006, 87_006, 0, 0)
    return {
        "message_rows": message_rows,
        "feature_store": checks,
        "target_position_cache": {
            "rows": target_positions[0],
            "unmatched": target_positions[2],
            "time_slot_mismatches": target_positions[3],
        },
    }


def audit_outputs() -> dict[str, object]:
    tables = SUBMISSION / "tables"
    frozen_hashes = {
        ROOT / "artifacts/models/target_relationship_rescue/model.joblib": "38bb3c3d59eb178b6ce77940ab04ef64b72cbb88ad9008cfa7bd7fd86b4d9591",
        ROOT / "artifacts/models/target_relationship_rescue/freeze_manifest.json": "5d1913a6149f54146c7698b5b03aa799c94ac30daa9613e4565ce0269f83e0cd",
        ROOT / "artifacts/tables/target_relationship_june_predictions.parquet": "e1e44f0c3d5cce2d1aba0452953b8965dc04ea4d459fe345a77e6d7e009b4dea",
        tables / "target_relationship_validation.json": "4ba914a83e6169cd6d893938eaa11f477869cb4d0911555966552e68e641c083",
        tables / "target_relationship_june_reporting.json": "33f9cd29ecbc34d1b87e2dcb44e2844bfe8a8a31e86199c19fde1c4303afbc5b",
    }
    for path, expected in frozen_hashes.items():
        assert path.exists(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    recipe = json.loads(REPRODUCTION_RECIPE.read_text())
    assert recipe["recipe_status"] == "FROZEN_REPRODUCTION_ONLY"
    assert recipe["training_window"] == {
        "start_inclusive": "2026-03-12T00:00:00Z",
        "end_exclusive": "2026-05-01T00:00:00Z",
        "start_unix": 1_773_273_600,
        "end_unix": 1_777_593_600,
    }
    assert recipe["negative_sampling"]["stride"] == 2
    assert recipe["negative_sampling"]["sampled_negative_weight"] == 2.0
    assert recipe["lightgbm_parameters"]["random_state"] == 20260811
    assert recipe["lightgbm_parameters"]["n_estimators"] == 500
    assert recipe["threshold"] == 0.23211809647507783
    assert recipe["verification"]["model_serialization"]["byte_stable"] is False

    for path in (
        REPRODUCTION_FEATURES,
        REPRODUCTION_FEATURE_MANIFEST,
        REPRODUCTION_MODEL,
        REPRODUCTION_PREDICTIONS,
        REPRODUCTION_REPORT,
    ):
        assert path.exists(), path
    reproduction_report = json.loads(REPRODUCTION_REPORT.read_text())
    assert reproduction_report["status"] == "PASS_FRESH_TRAINING_AND_JUNE_SCORING"
    assert reproduction_report["canonical_promoted_artifacts_written"] is False
    assert reproduction_report["bounded_result_correction"] == recipe["bounded_correction"]
    fresh_predictions = pd.read_parquet(REPRODUCTION_PREDICTIONS).sort_values(
        ["block_time", "token_address"], kind="mergesort"
    )
    frozen_predictions = pd.read_parquet(
        ROOT / "artifacts/tables/target_relationship_june_predictions.parquet"
    ).sort_values(["block_time", "token_address"], kind="mergesort")
    assert fresh_predictions[["token_address", "block_time", "label"]].reset_index(
        drop=True
    ).equals(
        frozen_predictions[["token_address", "block_time", "label"]].reset_index(
            drop=True
        )
    )
    score_error = (
        fresh_predictions.score.to_numpy()
        - frozen_predictions.score.to_numpy()
    )
    max_score_error = float(abs(score_error).max())
    assert max_score_error <= recipe["verification"][
        "full_score_absolute_tolerance_local_audit"
    ]
    selection_differences = int(
        (
            fresh_predictions.selected.to_numpy()
            != frozen_predictions.selected.to_numpy()
        ).sum()
    )
    assert selection_differences == 0
    classification = json.loads((tables / "classification_metrics.json").read_text())
    active = json.loads((tables / "active_period_training.json").read_text())
    backtest = json.loads((tables / "backtest_results.json").read_text())
    intraslot = json.loads((tables / "intraslot_latency_sensitivity.json").read_text())
    robustness = json.loads((tables / "robustness_summary.json").read_text())
    historical = json.loads((tables / "historical_outcome_audit.json").read_text())
    developer_sell = json.loads(
        (tables / "developer_sell_outcome_results.json").read_text()
    )
    ranking = json.loads((tables / "ranking_hard_negative_results.json").read_text())
    strategy = json.loads((tables / "profitable_disagreement_results.json").read_text())
    curve = json.loads((tables / "curve_replay_results.json").read_text())
    raw_block = json.loads((tables / "targeted_raw_block_audit.json").read_text())
    availability = json.loads((tables / "third_pass_availability_audit.json").read_text())
    final_importance = pd.read_csv(tables / "final_selector_feature_importance.csv")
    final_effects = pd.read_csv(tables / "final_selector_feature_effects.csv")
    relationship_validation = json.loads(
        (tables / "target_relationship_validation.json").read_text()
    )
    relationship_june = json.loads(
        (tables / "target_relationship_june_reporting.json").read_text()
    )
    relationship_importance = pd.read_csv(
        tables / "target_relationship_feature_importance.csv"
    )
    relationship_summary = pd.read_csv(
        tables / "target_relationship_classification_summary.csv"
    )
    behavior = json.loads((tables / "behavior_summary.json").read_text())
    head_to_head = pd.read_csv(tables / "third_pass_head_to_head.csv")
    final = classification["final_test"]
    active_final = active["active_era_model"]["june"]
    for key in ("rows", "positives", "pr_auc", "precision", "recall", "f1", "predicted_entries"):
        assert final[key] == active_final[key], key
    promoted_april = relationship_validation["windows"]["april"][
        "final_plus_expanded_relationship"
    ]
    promoted_may = relationship_validation["windows"]["may"][
        "final_plus_expanded_relationship"
    ]
    promoted_june = relationship_june["metrics"]
    assert promoted_april["pr_auc"] == 0.2820629224833508
    assert promoted_may["pr_auc"] == 0.38599924431037996
    assert promoted_june["pr_auc"] == 0.20471037711709195
    assert promoted_june["predicted_entries"] == 6_094
    assert promoted_june["true_positives"] == 1_787
    assert relationship_summary.population.tolist() == ["April", "May", "June"]
    assert all(
        math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15)
        for observed, expected in zip(
            relationship_summary.pr_auc,
            (
                promoted_april["pr_auc"],
                promoted_may["pr_auc"],
                promoted_june["pr_auc"],
            ),
            strict=True,
        )
    )
    assert relationship_summary.loc[relationship_summary.population.eq("June"), "fit_end"].item() == "2026-05-01 exclusive"
    assert relationship_validation["decision_clock"] == (
        "A raw target buy contributes only when target_buy_time < candidate block_time; equality is excluded."
    )
    overlap = backtest["selection_overlap"]
    assert overlap["replica_entries"] == final["predicted_entries"]
    assert overlap["overlap"] == final["true_positives"]
    assert overlap["precision_vs_target"] == final["precision"]
    assert overlap["recall_of_target"] == final["recall"]
    expected_policies = {
        "offset_1",
        "offset_10",
        "offset_25",
        "offset_50",
        "offset_100",
        "offset_118",
        "offset_150",
        "offset_250",
        "empirical_pre_june_same_slot",
    }
    assert set(intraslot["results"]) == expected_policies
    assert robustness["top10_stability"]["intersection_count"] == 8
    assert historical["pre_june_decision"]["status"] == "KEEP"
    assert historical["windows"]["april"]["daily_pr_auc_stability"]["positive_delta_share"] > 0.6
    assert historical["windows"]["may"]["daily_pr_auc_stability"]["positive_delta_share"] > 0.6
    assert historical["temporal_audit"] == {
        "rows": 5_076_421,
        "tokens": 5_076_421,
        "duplicate_keys": 0,
        "future_launches": 0,
        "future_states": 0,
        "invalid_claim_recencies": 0,
        "invalid_1d_fractions": 0,
        "invalid_7d_fractions": 0,
        "invalid_bounded_fractions": 0,
        "incomplete_launch_histories": 1_439,
    }
    assert developer_sell["pre_june_decision"] == {
        "april_incremental_pr_auc": 0.0008308540024735966,
        "may_incremental_pr_auc": 0.0006652269018083135,
        "rule": "Add developer-sell features only for >=0.002 May PR-AUC lift over creator-fee quality and no material (>0.0005) April regression.",
        "status": "DROP",
    }
    assert developer_sell["temporal_audit"] == {
        "rows": 5_076_421,
        "tokens": 5_076_421,
        "duplicate_keys": 0,
        "future_launches": 0,
        "future_states": 0,
        "invalid_recencies": 0,
        "invalid_fractions": 0,
        "invalid_latest_group_summaries": 0,
        "tied_latest_group_rows": 268_857,
    }
    assert all(
        decision["status"] == "DROP"
        for decision in ranking["pre_june_decisions"].values()
    )
    assert strategy["pre_june_decision"]["status"] == "KEEP"
    assert strategy["pre_june_strategy_operating_point"]["status"] == "KEEP"
    selective = strategy["june_reporting_only"]["selection"]["selective_two_stage"]
    assert selective["entries"] == 1_402 and selective["target_overlap"] == 205
    selective_curve = curve["results"]["offset_118"]["selective_two_stage"]
    assert selective_curve["fixed_quote"]["fee_0.0095"][
        "p99_capped_total_pnl_sol_supported"
    ] < 0
    assert selective_curve["fixed_token"]["fee_0.0095"][
        "p99_capped_total_pnl_sol_supported"
    ] > 0
    assert raw_block["decoded_trade_events"] == 82
    assert raw_block["normalization_audit"]["max_sol_error"] == 0
    assert raw_block["normalization_audit"]["max_token_error"] == 0
    assert availability["graph_edge_audit"]["total_nonempty_from"] == 0
    assert availability["graph_edge_audit"]["total_nonempty_to"] == 0
    expected_control_top10 = [
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
    assert final_importance["rank"].tolist() == list(range(1, len(final_importance) + 1))
    assert final_importance.head(10)["feature"].tolist() == expected_control_top10
    assert set(final_effects["feature"]) == set(expected_control_top10)
    expected_promoted_top10 = [
        "deployments_since_prior_target_buy",
        "seconds_since_prior_deploy",
        "dev_buy_sol",
        "prior_target_buy_rate_30d",
        "prior_target_buy_rate_7d",
        "seconds_since_activity",
        "prior_deploy_count_7d",
        "days_since_2026_start",
        "hist_claimed_launch_fraction",
        "compute_unit_price_micro_lamports",
    ]
    assert relationship_importance["rank"].tolist() == list(
        range(1, len(relationship_importance) + 1)
    )
    assert relationship_importance.head(10).feature.tolist() == expected_promoted_top10
    assert math.isclose(relationship_importance.gain_share.sum(), 1.0)
    assert set(head_to_head["policy"]) == {
        "target_equal_stake_counterfactual",
        "baseline",
        "quality_augmented",
        "two_stage",
        "selective_two_stage",
    }
    target_row = head_to_head.set_index("policy").loc[
        "target_equal_stake_counterfactual"
    ]
    target_cashflow = backtest["actual_target_cashflow_june"]
    assert target_row["june_entries"] == 4_195
    assert target_row["actual_target_cashflow_fully_fee_adjusted_pnl_usd"] == target_cashflow[
        "fully_fee_adjusted_pnl_usd"
    ]
    assert target_cashflow["fully_fee_adjusted_pnl_usd"] == 185_610.17369310948
    cashflow = behavior["cashflow_performance_bought_positions"]
    assert cashflow["fully_fee_adjusted_pnl_usd"] == 925_055.7199093377
    assert cashflow["fully_fee_adjusted_hit_rate"] == 0.5864629091134071
    assert math.isclose(
        cashflow["total_defensible_cost_usd"],
        cashflow["network_execution_cost_usd"] + cashflow["pump_separate_cost_usd"],
    )
    token_metrics = pd.read_parquet(TOKEN_METRICS)
    bought_metrics = token_metrics[token_metrics.buy_transactions.gt(0)]
    assert len(bought_metrics) == cashflow["scope_bought_positions"]
    assert (bought_metrics.fees_usd == bought_metrics.total_defensible_cost_usd).all()
    assert math.isclose(
        bought_metrics.net_pnl_usd.sum(), cashflow["fully_fee_adjusted_pnl_usd"]
    )
    assert math.isclose(
        pd.read_csv(tables / "behavior_monthly.csv").fully_fee_adjusted_pnl_usd.sum(),
        cashflow["fully_fee_adjusted_pnl_usd"],
    )
    assert backtest["strategy"]["marginal_accounting"] == (
        "network-cost-adjusted and gross of proportional Pump swap fees"
    )
    assert curve["frozen_strategy"]["round_trip_network_fee_sol"] == 0.09101000000000001
    assert curve["bounds"]["observed_fee_rate_range"] == [0.0095, 0.0125]

    curve_policy_names = {
        "target_equal_stake_counterfactual": "target_equal_stake",
        "baseline": "baseline_replica",
        "quality_augmented": "quality_augmented_replica",
        "two_stage": "two_stage",
        "selective_two_stage": "selective_two_stage",
    }
    indexed_head_to_head = head_to_head.set_index("policy")
    for table_policy, curve_policy in curve_policy_names.items():
        table_row = indexed_head_to_head.loc[table_policy]
        curve_row = curve["results"]["offset_118"][curve_policy]
        for intent, prefix in (
            ("fixed_quote", "curve_fixed_quote"),
            ("fixed_token", "curve_fixed_token"),
        ):
            metrics = curve_row[intent]["fee_0.0095"]
            assert math.isclose(
                table_row[f"{prefix}_coverage"],
                metrics["coverage"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            assert math.isclose(
                table_row[f"{prefix}_median_roi"],
                metrics["median_net_roi"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            assert math.isclose(
                table_row[f"{prefix}_p99_capped_pnl_sol"],
                metrics["p99_capped_total_pnl_sol_supported"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )

    writeup = (SUBMISSION / "writeup.md").read_text()
    assert len(writeup.split()) < 3_000
    expected_snippets = [
        f"{promoted_june['pr_auc']:.10f}",
        f"{promoted_june['precision']:.2%}",
        f"{promoted_june['recall']:.2%}",
        f"{promoted_june['f1']:.5f}",
        f"{promoted_june['predicted_entries']:,}",
        f"{intraslot['results']['offset_118']['replica']['median_roi']:.2%}".replace("-", "−"),
        f"{promoted_april['pr_auc']:.6f}",
        f"{promoted_may['pr_auc']:.6f}",
        f"{selective['entries']:,}",
        "70,805",
        "+$117.55 / −$28.30",
        "58.65%",
        "34.93% / −10.71%",
        "+$185,610.17",
        "+$925,056",
        "2026-03-12 inclusive through 2026-05-01 exclusive",
        "target_buy_time < current candidate block_time",
        "gross of swap fees",
        "conditional ROI need not decline monotonically",
    ]
    for snippet in expected_snippets:
        assert snippet in writeup, snippet

    def pct(value: float) -> str:
        return f"{value:.2%}".replace("-", "−")

    def pnl(value: float) -> str:
        return f"{value:+.1f}".replace("-", "−")

    writeup_curve_policies = {
        "Preserved control": "baseline_replica",
        "Prior quality selector": "quality_augmented_replica",
        "Equal-count two-stage": "two_stage",
        "Selective two-stage": "selective_two_stage",
    }
    for label, curve_policy in writeup_curve_policies.items():
        exact = curve["results"]["offset_118"][curve_policy]
        quote = exact["fixed_quote"]["fee_0.0095"]
        token = exact["fixed_token"]["fee_0.0095"]
        assert label in writeup
        assert (
            f"{quote['coverage'] * 100:.2f}–{pct(token['coverage'])}" in writeup
        )
        assert f"{pct(quote['median_net_roi'])} to {pct(token['median_net_roi'])}" in writeup
        assert (
            f"{pnl(quote['p99_capped_total_pnl_sol_supported'])} to "
            f"{pnl(token['p99_capped_total_pnl_sol_supported'])} SOL"
        ) in writeup
    selective_upper_fee = curve["results"]["offset_118"]["selective_two_stage"]
    assert (
        f"{pnl(selective_upper_fee['fixed_quote']['fee_0.0125']['p99_capped_total_pnl_sol_supported'])} "
        f"to {pnl(selective_upper_fee['fixed_token']['fee_0.0125']['p99_capped_total_pnl_sol_supported'])} SOL"
    ) in writeup

    notebook = json.loads((SUBMISSION / "final_notebook.ipynb").read_text())
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "/home/" not in sources
    assert "KAGGLE_" not in sources
    assert "/kaggle/" not in sources
    assert "## 6. Primary Part 3: source-built marginal-price backtest" in sources
    assert "gross of proportional Pump swap fees" in sources
    assert "## 7. Exact Pump mechanics and bounded conclusions" in sources
    assert "reproduce_frozen_exact_replay" in sources
    assert "PASS_FRESH_STRATEGY_SELECTION_AND_EXACT_CURVE_REPLAY" in sources
    assert "frozen_part2_reproduction_features.parquet" not in sources
    assert "third_pass_june_strategy_predictions.parquet" not in sources
    assert "june_curve_replay_outcomes.parquet" not in sources
    assert "symlink_to" not in sources
    assert "data/raw/" in sources
    assert "SOLANA_BOUGHT_TXS" in sources
    assert "SOLANA_JUNE_TRADES" in sources
    for module in (
        "behavior",
        "frozen_reproduction",
    ):
        assert f"solana_sniper_reverse_engineering.{module}" in sources
    assert "run_frozen_reproduction" in sources
    assert "force_backtest=True" in sources
    assert "run_curve_replay" not in sources
    assert "target_relationship_rescue/model.joblib" not in sources
    assert "target_relationship_june_predictions.parquet" not in sources
    assert "target_relationship_feature_importance.csv" in sources
    assert "target_relationship_feature_dictionary.csv" in sources
    assert "curve_replay_results.json" in sources
    assert "developer_sell_outcome_results.json" not in sources
    assert "third_pass_head_to_head.csv" not in sources
    assert "methodological_audit.json" not in sources
    notebook_outputs = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    ]
    output_blob = json.dumps(notebook_outputs)
    embedded_pngs = sum(
        "image/png" in output.get("data", {}) for output in notebook_outputs
    )
    # The generated submission notebook stays unexecuted; the execution helper can save
    # outputs into a separate working copy when a judge or author runs it.
    assert embedded_pngs == 0
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert "FigureCanvasAgg" not in output_blob
    assert not re.search(r"/(?:home|Users)/[^/]+/", output_blob)
    assert not re.search(r"[A-Za-z]:\\Users\\[^\\]+", output_blob)

    assert "strict online memory of signers" in sources
    assert "Fresh June reproduction" in sources

    exact_report = json.loads(EXACT_REPLAY_REPORT.read_text())
    assert exact_report["status"] == "PASS_FRESH_STRATEGY_SELECTION_AND_EXACT_CURVE_REPLAY"
    assert exact_report["aggregate_reference_match"] is True
    stale_outcomes = pd.read_parquet(
        ROOT / "artifacts/superseded_20260813_tie_correction/exact_replay/june_curve_replay_outcomes.parquet",
        columns=["token_address"],
    )
    current_strategy = pd.read_parquet(
        ROOT / "artifacts/superseded_20260813_tie_correction/exact_replay/third_pass_june_strategy_predictions.parquet"
    )
    membership_columns = [
        "label",
        "baseline_selected",
        "quality_selected",
        "two_stage_selected",
        "selective_two_stage_selected",
    ]
    current_union = set(
        current_strategy.loc[
            current_strategy[membership_columns].eq(1).any(axis=1), "token_address"
        ]
    )
    stale_union = set(stale_outcomes.token_address)
    stale_extra_tokens = len(stale_union - current_union)
    stale_missing_tokens = len(current_union - stale_union)
    assert stale_extra_tokens == 1_133
    assert stale_missing_tokens == 1_141

    readme = (ROOT / "README.md").read_text()
    assert "docs/" not in readme
    assert "Competition data is **not included in this repository**" in readme
    assert "through the competition source" in readme
    public_resource = json.loads(
        (tables / "public_resource_manifest.json").read_text()
    )
    assert public_resource["publication_status"] == "READY_FOR_MANUAL_AGGREGATE_PERMISSION_REVIEW"
    assert public_resource["safe_to_publish_now"] is False
    assert len(public_resource["publication_blockers"]) == 1
    assert public_resource["public_resource_required_for_current_notebook"] is False
    assert public_resource["resource_role"] == "optional source-only publication bundle"
    assert "SOLANA_*" in public_resource["competition_input_policy"]
    assert "local ignored working artifacts" in public_resource["notebook_output_policy"]
    assert public_resource["contains_competition_raw_data"] is False
    assert public_resource["contains_row_level_competition_derivatives"] is False
    assert public_resource["contains_derived_feature_or_label_cache"] is False
    assert public_resource["contains_strategy_prediction_or_selection_cache"] is False
    assert public_resource["contains_exact_curve_outcome_cache"] is False
    assert public_resource["contains_trained_models"] is False
    assert public_resource["prohibited_files"] == []
    resource_paths = {item["path"] for item in public_resource["files"]}
    assert not any(path.endswith(".parquet") for path in resource_paths)
    assert not any("public_cache" in path for path in resource_paths)
    assert "submission/tables/targeted_raw_block_trade_events.csv" not in resource_paths
    assert public_resource["total_bytes"] < 2_000_000
    assert not any("target_relationship_rescue" in path for path in resource_paths)
    assert "public_cache/target_relationship_june_predictions.parquet" not in resource_paths
    return {
        "writeup_words": len(writeup.split()),
        "notebook_cells": len(notebook["cells"]),
        "preserved_control_classification": final,
        "promoted_classification": promoted_june,
        "selection_overlap": overlap,
        "historical_outcome_decision": historical["pre_june_decision"],
        "developer_sell_outcome_decision": developer_sell["pre_june_decision"],
        "historical_outcome_daily_stability": {
            period: historical["windows"][period]["daily_pr_auc_stability"]
            for period in ("april", "may")
        },
        "economic_operating_point": strategy["pre_june_strategy_operating_point"],
        "promoted_selector_top10": expected_promoted_top10,
        "head_to_head_policies": head_to_head["policy"].tolist(),
        "embedded_notebook_pngs": embedded_pngs,
        "exact_replay_reconstruction": {
            "status": exact_report["status"],
            "superseded_cache_extra_tokens": stale_extra_tokens,
            "superseded_cache_missing_tokens": stale_missing_tokens,
        },
        "raw_block_events": raw_block["decoded_trade_events"],
        "frozen_classifier_hashes": {
            str(path.relative_to(ROOT)): expected
            for path, expected in frozen_hashes.items()
        },
        "private_cache_reference_reproduction": {
            "scope": "local canonical/reference audit only; not evidence of clean public reconstruction",
            "status": reproduction_report["status"],
            "max_absolute_score_error_vs_frozen": max_score_error,
            "selection_differences_vs_frozen": selection_differences,
            "june_metrics": reproduction_report["june_metrics"],
            "part3_overlap": reproduction_report["part3_reproduction"][
                "selection_overlap"
            ],
        },
        "public_resource": {
            "publication_status": public_resource["publication_status"],
            "files": public_resource["file_count"],
            "bytes": public_resource["total_bytes"],
            "contains_promoted_model": public_resource[
                "contains_trained_models"
            ],
            "contains_promoted_predictions": public_resource[
                "contains_strategy_prediction_or_selection_cache"
            ],
        },
    }


def audit_git_hygiene() -> dict[str, object]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode().split("\0")
    tracked = [item for item in tracked if item]
    prohibited = [
        item
        for item in tracked
        if item.startswith(("data/raw/", "data/downloads/", "data/interim/", "data/processed/", "artifacts/"))
    ]
    prohibited_row_level_derivatives = [
        item
        for item in tracked
        if (ROOT / item).exists()
        if item.endswith(
            (
                "frozen_part2_reproduction_features.parquet",
                "june_curve_replay_outcomes.parquet",
                "targeted_raw_block_trade_events.csv",
                "third_pass_june_strategy_predictions.parquet",
            )
        )
    ]
    assert not prohibited, prohibited
    assert not prohibited_row_level_derivatives, prohibited_row_level_derivatives
    large = [
        item
        for item in tracked
        if (ROOT / item).exists() and (ROOT / item).stat().st_size > 10_000_000
    ]
    assert not large, large
    return {
        "tracked_files": len(tracked),
        "prohibited_tracked": prohibited,
        "prohibited_row_level_derivatives": prohibited_row_level_derivatives,
        "large_tracked": large,
    }


def isolated_positive_extraction() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="solana-message-audit-", dir="/tmp") as directory:
        output = Path(directory) / "bought.parquet"
        manifest = extract_file(BOUGHT_TXS, output, force=True)
        canonical = INTERIM / "message_features" / "bought.parquet"
        con = duckdb.connect()
        checks = []
        for path in (canonical, output):
            checks.append(
                con.execute(
                    """
                    SELECT count(*), count(DISTINCT (tx_hash, token_address)),
                           sum(hash(tx_hash, token_address, block_time, block_slot))::HUGEINT
                    FROM read_parquet(?)
                    """,
                    [str(path)],
                ).fetchone()
            )
        assert checks[0] == checks[1]
        assert checks[0][0] == int(manifest["token_row_count"])
        return {
            "source": str(BOUGHT_TXS.relative_to(ROOT)),
            "rows": checks[0][0],
            "distinct_keys": checks[0][1],
            "content_checksum": str(checks[0][2]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final reproducibility and publication hygiene")
    parser.add_argument("--isolated-positive-extraction", action="store_true")
    args = parser.parse_args()
    result = {
        "manifests": audit_manifests(),
        "outputs": audit_outputs(),
        "git_hygiene": audit_git_hygiene(),
    }
    if args.isolated_positive_extraction:
        result["isolated_positive_extraction"] = isolated_positive_extraction()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()