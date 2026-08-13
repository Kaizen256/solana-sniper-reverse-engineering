#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from solana_sniper_reverse_engineering.behavior import TARGET_POSITIONS
from solana_sniper_reverse_engineering.feature_store import (
    ACTIVITY_STATE,
    BASE_FEATURES,
    FEATURE_MANIFEST,
    FEATURE_STORE,
    audit as audit_features,
)
from solana_sniper_reverse_engineering.message_features import extract_file


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
    head_to_head = pd.read_csv(tables / "third_pass_head_to_head.csv")
    final = classification["final_test"]
    active_final = active["active_era_model"]["june"]
    for key in ("rows", "positives", "pr_auc", "precision", "recall", "f1", "predicted_entries"):
        assert final[key] == active_final[key], key
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
    assert developer_sell["pre_june_decision"]["status"] == "KEEP"
    assert developer_sell["pre_june_decision"]["may_incremental_pr_auc"] > 0.002
    assert developer_sell["temporal_audit"] == {
        "rows": 5_076_421,
        "tokens": 5_076_421,
        "duplicate_keys": 0,
        "future_launches": 0,
        "future_states": 0,
        "invalid_recencies": 0,
        "invalid_fractions": 0,
    }
    assert all(
        decision["status"] == "DROP"
        for decision in ranking["pre_june_decisions"].values()
    )
    assert strategy["pre_june_decision"]["status"] == "KEEP"
    assert strategy["pre_june_strategy_operating_point"]["status"] == "KEEP"
    selective = strategy["june_reporting_only"]["selection"]["selective_two_stage"]
    assert selective["entries"] == 1_121 and selective["target_overlap"] == 177
    selective_curve = curve["results"]["offset_118"]["selective_two_stage"]
    for intent in ("fixed_quote", "fixed_token"):
        assert selective_curve[intent]["fee_0.0095"]["p99_capped_total_pnl_sol_supported"] > 0
        assert selective_curve[intent]["fee_0.0125"]["p99_capped_total_pnl_sol_supported"] > 0
    assert raw_block["decoded_trade_events"] == 82
    assert raw_block["normalization_audit"]["max_sol_error"] == 0
    assert raw_block["normalization_audit"]["max_token_error"] == 0
    assert availability["graph_edge_audit"]["total_nonempty_from"] == 0
    assert availability["graph_edge_audit"]["total_nonempty_to"] == 0
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
    assert final_importance["rank"].tolist() == list(range(1, len(final_importance) + 1))
    assert final_importance.head(10)["feature"].tolist() == expected_top10
    assert set(final_effects["feature"]) == set(expected_top10)
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
    assert target_row["actual_target_cashflow_net_pnl_usd"] == target_cashflow[
        "net_pnl_usd"
    ]
    assert target_cashflow["net_pnl_usd"] == 213_884.4139812971

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
        f"{final['pr_auc']:.4f}",
        f"{final['precision']:.2%}",
        f"{final['recall']:.2%}",
        f"{final['f1']:.5f}",
        f"{final['predicted_entries']:,}",
        f"{intraslot['results']['offset_118']['replica']['median_roi']:.2%}".replace("-", "−"),
        f"{developer_sell['windows']['may']['creator_fee_plus_developer_sell']['pr_auc']:.5f}",
        f"{developer_sell['june_reporting_only']['creator_fee_plus_developer_sell']['pr_auc']:.5f}",
        f"{selective['entries']:,}",
        "70,805",
        "+$116.01 / −$27.06",
        "0.03692",
        "48.61% / −2.06%",
        "+$213,884",
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
        "Final imitation": "quality_augmented_replica",
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
    assert "## 8. Marginal +118 head-to-head" in sources
    assert "not exact curve replay" in sources.lower()
    assert "## 9. Exact Pump curve execution bounds" in sources
    assert "run_curve_replay" in sources
    assert "curve_replay_results.json" in sources
    assert "curve_replay_coverage.csv" in sources
    for module in (
        "behavior",
        "message_features",
        "feature_store",
        "modeling",
        "backtest",
        "robustness",
        "methodological_audit",
    ):
        assert f"solana_sniper_reverse_engineering.{module}" in sources
    assert "historical_outcome_audit.json" in sources
    assert "developer_sell_outcome_results.json" in sources
    assert "final_selector_feature_importance.csv" in sources
    assert "final_selector_feature_effects.csv" in sources
    assert "third_pass_head_to_head.csv" in sources
    notebook_outputs = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    ]
    output_blob = json.dumps(notebook_outputs)
    embedded_pngs = sum(
        "image/png" in output.get("data", {}) for output in notebook_outputs
    )
    assert embedded_pngs == 5
    assert "FigureCanvasAgg" not in output_blob
    assert not re.search(r"/(?:home|Users)/[^/]+/", output_blob)
    assert not re.search(r"[A-Za-z]:\\\\Users\\\\[^\\\\]+", output_blob)

    exact_heading_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "".join(cell.get("source", [])).startswith("## 9. Exact Pump")
    )
    exact_output_blob = json.dumps(
        notebook["cells"][exact_heading_index + 1].get("outputs", [])
    )
    for curve_policy in writeup_curve_policies.values():
        exact = curve["results"]["offset_118"][curve_policy]
        for intent in ("fixed_quote", "fixed_token"):
            for fee in ("fee_0.0095", "fee_0.0125"):
                metrics = exact[intent][fee]
                assert f"{metrics['coverage']:.2%}" in exact_output_blob
                assert f"{metrics['median_net_roi']:.2%}" in exact_output_blob
                assert (
                    f"{metrics['p99_capped_total_pnl_sol_supported']:+.1f}"
                    in exact_output_blob
                )

    readme = (ROOT / "README.md").read_text()
    assert "docs/" not in readme
    assert "Competition data is **not included in this repository**" in readme
    assert "through the competition source" in readme
    return {
        "writeup_words": len(writeup.split()),
        "notebook_cells": len(notebook["cells"]),
        "final_classification": final,
        "selection_overlap": overlap,
        "historical_outcome_decision": historical["pre_june_decision"],
        "developer_sell_outcome_decision": developer_sell["pre_june_decision"],
        "historical_outcome_daily_stability": {
            period: historical["windows"][period]["daily_pr_auc_stability"]
            for period in ("april", "may")
        },
        "economic_operating_point": strategy["pre_june_strategy_operating_point"],
        "final_selector_top10": expected_top10,
        "head_to_head_policies": head_to_head["policy"].tolist(),
        "embedded_notebook_pngs": embedded_pngs,
        "raw_block_events": raw_block["decoded_trade_events"],
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
    assert not prohibited, prohibited
    large = [
        item
        for item in tracked
        if (ROOT / item).exists() and (ROOT / item).stat().st_size > 10_000_000
    ]
    assert not large, large
    return {"tracked_files": len(tracked), "prohibited_tracked": prohibited, "large_tracked": large}


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
