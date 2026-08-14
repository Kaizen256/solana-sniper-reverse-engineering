#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import joblib
import matplotlib
import nbformat as nbf
import numpy as np
import pandas as pd

from solana_sniper_reverse_engineering.config import (
    BOUGHT_ACTIVITY,
    BOUGHT_INDEX,
    BOUGHT_TXS,
    JITO_TRANSACTIONS,
    JUNE_TRADES,
    NOT_BOUGHT_ACTIVITY,
    NOT_BOUGHT_INDEX,
    NOT_BOUGHT_TXS,
    TARGET_ACTIVITY,
    TARGET_TXS,
    TARGET_TX_INDEX,
)
from solana_sniper_reverse_engineering.frozen_reproduction import (
    strategy_selection_fingerprints,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"


def _prediction_fingerprints(predictions: pd.DataFrame, threshold: float) -> dict[str, str]:
    ordered = predictions.sort_values(["block_time", "token_address"], kind="mergesort")
    score = ordered.score.to_numpy(dtype=np.float64)
    labels = ordered.label.to_numpy(dtype=np.uint8)
    selected = score >= threshold
    quantized = np.rint(score * 100_000_000_000).astype("<i8", copy=False)
    rank_order = np.lexsort(
        (ordered.token_address.astype(str).to_numpy(), -score)
    )
    ranking_digest = hashlib.sha256()
    for index in rank_order:
        ranking_digest.update(str(ordered.token_address.iloc[index]).encode())
        ranking_digest.update(b"\n")
    selected_digest = hashlib.sha256()
    label_selection_digest = hashlib.sha256()
    for token, label, is_selected in zip(
        ordered.token_address.astype(str), labels, selected, strict=True
    ):
        if is_selected:
            selected_digest.update(token.encode())
            selected_digest.update(b"\n")
        label_selection_digest.update(token.encode())
        label_selection_digest.update(bytes((int(label), int(is_selected))))
    return {
        "score_quantized_11_sha256": hashlib.sha256(quantized.tobytes()).hexdigest(),
        "full_ranking_sha256": ranking_digest.hexdigest(),
        "selected_token_sha256": selected_digest.hexdigest(),
        "token_label_selection_sha256": label_selection_digest.hexdigest(),
    }


def build_public_reproduction_recipe() -> None:
    """Publish the complete immutable recipe without publishing a fitted model.

    The existing ignored freeze manifest is left byte-for-byte unchanged. This tracked
    derivative adds the preprocessing, LightGBM, sampling, and deterministic reference
    fingerprints that a public notebook needs to perform and verify a fresh fit.
    """
    model_dir = ROOT / "artifacts" / "models" / "target_relationship_rescue"
    frozen_manifest_path = model_dir / "freeze_manifest.json"
    frozen_model_path = model_dir / "model.joblib"
    frozen_predictions_path = (
        ROOT / "artifacts" / "tables" / "target_relationship_june_predictions.parquet"
    )
    freeze = json.loads(frozen_manifest_path.read_text())
    pipeline = joblib.load(frozen_model_path)
    estimator = pipeline.named_steps["model"]
    threshold = float(freeze["threshold_selected_on_may"])
    june_report = json.loads(
        (SUBMISSION / "tables" / "target_relationship_june_reporting.json").read_text()
    )
    predictions = pd.read_parquet(frozen_predictions_path)
    required_paths = (
        ("SOLANA_BOUGHT_TXS", BOUGHT_TXS),
        ("SOLANA_BOUGHT_INDEX", BOUGHT_INDEX),
        ("SOLANA_BOUGHT_ACTIVITY", BOUGHT_ACTIVITY),
        ("SOLANA_NOT_BOUGHT_TXS", NOT_BOUGHT_TXS),
        ("SOLANA_NOT_BOUGHT_INDEX", NOT_BOUGHT_INDEX),
        ("SOLANA_NOT_BOUGHT_ACTIVITY", NOT_BOUGHT_ACTIVITY),
        ("SOLANA_TARGET_ACTIVITY", TARGET_ACTIVITY),
        ("SOLANA_TARGET_TXS", TARGET_TXS),
        ("SOLANA_TARGET_TX_INDEX", TARGET_TX_INDEX),
        ("SOLANA_JITO_TRANSACTIONS", JITO_TRANSACTIONS),
        ("SOLANA_JUNE_TRADES", JUNE_TRADES),
    )
    control_meta = json.loads((ROOT / "artifacts/models/model_features.json").read_text())
    control_operating_point = json.loads(
        (ROOT / "artifacts/models/operating_point.json").read_text()
    )
    quality_model_dir = ROOT / "artifacts/models/quality_augmented_final"
    quality_meta = json.loads((quality_model_dir / "model_features.json").read_text())
    quality_operating_point = json.loads(
        (quality_model_dir / "operating_point.json").read_text()
    )
    exact_strategy_predictions_path = (
        ROOT / "artifacts/reproduction/frozen_exact_strategy_predictions.parquet"
    )
    if not exact_strategy_predictions_path.exists():
        raise FileNotFoundError(
            "Rebuild the private source-refit exact strategy selections before "
            "freezing the public recipe"
        )
    strategy_predictions = pd.read_parquet(exact_strategy_predictions_path)
    recipe = {
        "recipe_status": "FROZEN_REPRODUCTION_ONLY",
        "purpose": "Train a fresh reproduction copy; never overwrite the canonical promoted artifact.",
        "freeze_reference": {
            "freeze_manifest_sha256": hashlib.sha256(
                frozen_manifest_path.read_bytes()
            ).hexdigest(),
            "canonical_model_joblib_sha256": hashlib.sha256(
                frozen_model_path.read_bytes()
            ).hexdigest(),
            "canonical_june_prediction_parquet_sha256": hashlib.sha256(
                frozen_predictions_path.read_bytes()
            ).hexdigest(),
        },
        "training_window": {
            "start_inclusive": "2026-03-12T00:00:00Z",
            "end_exclusive": "2026-05-01T00:00:00Z",
            "start_unix": 1_773_273_600,
            "end_unix": 1_777_593_600,
        },
        "june_reporting_window": {
            "start_inclusive": "2026-06-01T00:00:00Z",
            "end_exclusive": "2026-07-01T00:00:00Z",
            "start_unix": 1_780_272_000,
            "end_unix": 1_782_864_000,
        },
        "negative_sampling": {
            "procedure": "retain every positive and a negative iff DuckDB hash(token_address) % 2 = 0",
            "stride": 2,
            "positive_sample_weight": 1.0,
            "sampled_negative_weight": 2.0,
            "row_order": "block_time, token_address",
        },
        "preprocessing": {
            "numeric_imputer": {"strategy": "median", "add_indicator": True},
            "categorical_imputer": {"strategy": "most_frequent"},
            "one_hot_encoder": {
                "handle_unknown": "ignore",
                "sparse_output": False,
                "dtype": "float32",
            },
        },
        "lightgbm_parameters": estimator.get_params(),
        "features": {
            "numeric": freeze["numeric_features"],
            "categorical": freeze["categorical_features"],
            "selected_relationship": freeze["selected_relationship_features"],
            "developer_sell_tie_contract": freeze["developer_sell_tie_contract"],
        },
        "threshold": threshold,
        "threshold_policy": "fixed pre-June max-F1 threshold selected once on full May; never reselected here",
        "expected": {
            "feature_state": {
                "rows": 2_145_670,
                "tokens": 2_145_670,
                "duplicate_tokens": 0,
                "training_population_rows": 1_293_587,
                "training_population_positives": 6_655,
                "sampled_training_rows": 650_194,
                "june_rows": 852_083,
                "june_positives": 4_195,
            },
            "june_metrics": {
                key: june_report["metrics"][key]
                for key in (
                    "rows",
                    "positives",
                    "prevalence",
                    "pr_auc",
                    "pr_auc_lift_over_prevalence",
                    "roc_auc",
                    "threshold",
                    "precision",
                    "precision_lift_over_prevalence",
                    "recall",
                    "f1",
                    "predicted_entries",
                    "predicted_entry_rate",
                    "true_positives",
                )
            },
            "prediction_fingerprints": _prediction_fingerprints(
                predictions, threshold
            ),
            "frozen_model_string_sha256": hashlib.sha256(
                estimator.booster_.model_to_string().encode()
            ).hexdigest(),
        },
        "primary_part3_reference": {
            "role": "Aggregate-only reference for the source-built backtest fed by fresh corrected Part 2 selections.",
            "path": "submission/tables/target_relationship_primary_backtest.json",
            "sha256": hashlib.sha256(
                (SUBMISSION / "tables" / "target_relationship_primary_backtest.json").read_bytes()
            ).hexdigest(),
        },
        "verification": {
            "model_serialization": {
                "byte_stable": False,
                "evidence": "Repeated refits on the identical host and package stack changed last-bit leaf statistics/equivalent tie ordering, while full June scores differed by at most 1.67e-15 and selected membership/metrics were identical.",
                "policy": "Do not require serialized model equality. Require the exact full-ranking fingerprint, full 11-decimal calibration-vector fingerprint, exact selected-token fingerprints/counts, and strict metrics; the local audit additionally requires max absolute agreement to the frozen score vector <=1e-12. Eleven decimals avoids a brittle rounding-boundary failure observed at 6.11e-16 absolute score error.",
            },
            "full_score_absolute_tolerance_local_audit": 1e-12,
            "metric_absolute_tolerance": 1e-12,
            "selection_differences_allowed": 0,
        },
        "authorized_inputs": [
            {
                "environment_variable": environment_variable,
                "relative_path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
            }
            for environment_variable, path in required_paths
        ],
        "exact_replay_reproduction": {
            "status": "FROZEN_REPLAY_REPRODUCTION_ONLY",
            "role": "Secondary analysis: refit the corrected pre-June control, prior-quality, and economic models from gated inputs; regenerate June membership; then recompute integer curve outcomes. Row-level products remain private notebook outputs.",
            "control_model": {
                "training_start_inclusive": "2026-03-12T00:00:00Z",
                "training_end_exclusive": "2026-05-01T00:00:00Z",
                "negative_stride": 2,
                "expected_training_rows": 650_194,
                "numeric_features": control_meta["numeric_features"],
                "categorical_features": control_meta["categorical_features"],
                "threshold": control_operating_point["threshold"],
            },
            "quality_model": {
                "training_start_inclusive": "2026-03-12T00:00:00Z",
                "training_end_exclusive": "2026-05-01T00:00:00Z",
                "negative_stride": 2,
                "expected_training_rows": 650_194,
                "numeric_features": quality_meta["numeric_features"],
                "categorical_features": quality_meta["categorical_features"],
                "threshold": quality_operating_point["threshold"],
            },
            "economic_model": {
                "target": "creator_fee_claim_events_7d > 0",
                "training_start_unix": 1_773_273_600,
                "training_end_exclusive_unix": 1_779_667_200,
                "training_end_exclusive": "2026-05-25T00:00:00Z",
                "negative_stride": 5,
                "expected_training_rows": 404_676,
                "expected_training_positives": 9_074,
            },
            "two_stage_gate_multiplier": 2.0,
            "selective_trade_count_multiplier": 0.25,
            "expected_june_rows": 852_083,
            "expected_selection_fingerprints": strategy_selection_fingerprints(
                strategy_predictions
            ),
            "model_parameters": estimator.get_params(),
            "replay_reference": {
                "aggregate_results_path": "submission/tables/curve_replay_results.json",
                "aggregate_results_sha256": hashlib.sha256(
                    (SUBMISSION / "tables/curve_replay_results.json").read_bytes()
                ).hexdigest(),
                "row_level_predictions_public": False,
                "row_level_outcomes_public": False,
            },
        },
        "public_resource_policy": {
            "allowed": "source code, dependency configuration, frozen recipes, aggregate outputs, and figures only",
            "prohibited": "competition raw data; row-level competition-derived features, labels, scores, selections, or replay outcomes; trained models",
        },
        "bounded_correction": "The four ambiguous latest-prior developer-sell fields use the order-invariant latest-prior-launch-group-v1 contract. Organizer string-valued monetary history is accumulated at canonical nanounit precision for deterministic source rebuilds. The unchanged pre-June developer-sell gate then returns DROP, so its family is mechanically excluded downstream. No new feature family, LightGBM setting, threshold policy, or selection rule was introduced.",
    }
    (SUBMISSION / "tables" / "target_relationship_reproduction_recipe.json").write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\n"
    )


def build_promoted_selector_explainability() -> None:
    """Extract gain importance from the already-frozen promoted model.

    Loading tree metadata does not fit, score, select, threshold, or open June.
    The older permutation/effect tables remain in the package as documented
    controls, but figure 05 and the new table describe the promoted selector.
    """
    model_dir = ROOT / "artifacts" / "models" / "target_relationship_rescue"
    manifest = json.loads((model_dir / "freeze_manifest.json").read_text())
    model = joblib.load(model_dir / "model.joblib")
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    gain = estimator.booster_.feature_importance(importance_type="gain")
    splits = estimator.booster_.feature_importance(importance_type="split")
    relationship = set(manifest["selected_relationship_features"])
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": gain,
            "splits": splits,
        }
    ).sort_values(["gain", "feature"], ascending=[False, True], ignore_index=True)
    importance.insert(0, "rank", range(1, len(importance) + 1))
    importance["gain_share"] = importance.gain / importance.gain.sum()
    importance["feature_family"] = importance.feature.map(
        lambda feature: (
            "strict-prior target-signer relationship"
            if feature in relationship
            else "preserved signed-message/deployer-history foundation"
        )
    )
    importance.to_csv(
        SUBMISSION / "tables" / "target_relationship_feature_importance.csv",
        index=False,
    )

    validation = json.loads(
        (SUBMISSION / "tables" / "target_relationship_validation.json").read_text()
    )
    june = json.loads(
        (SUBMISSION / "tables" / "target_relationship_june_reporting.json").read_text()
    )
    classification_rows = []
    for population, role, fit_end, metrics in (
        (
            "April",
            "promotion gate",
            "2026-04-01 exclusive",
            validation["windows"]["april"]["final_plus_expanded_relationship"],
        ),
        (
            "May",
            "final promotion and threshold selection; labels not fitted",
            "2026-05-01 exclusive",
            validation["windows"]["may"]["final_plus_expanded_relationship"],
        ),
        (
            "June",
            "frozen reporting only",
            "2026-05-01 exclusive",
            june["metrics"],
        ),
    ):
        classification_rows.append(
            {
                "population": population,
                "role": role,
                "fit_start_inclusive": "2026-03-12",
                "fit_end": fit_end,
                **{
                    key: metrics[key]
                    for key in (
                        "rows",
                        "positives",
                        "prevalence",
                        "pr_auc",
                        "precision",
                        "recall",
                        "f1",
                        "predicted_entries",
                        "true_positives",
                        "threshold",
                    )
                },
            }
        )
    pd.DataFrame(classification_rows).to_csv(
        SUBMISSION / "tables" / "target_relationship_classification_summary.csv",
        index=False,
    )
    promoted = [
        validation["windows"][period]["final_plus_expanded_relationship"]["pr_auc"]
        for period in ("april", "may")
    ]
    control = [
        validation["windows"][period]["preserved_final"]["pr_auc"]
        for period in ("april", "may")
    ]
    top = importance.head(10).sort_values("gain")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = top.feature_family.map(
        {
            "strict-prior target-signer relationship": "#E07A2D",
            "preserved signed-message/deployer-history foundation": "#176B87",
        }
    )
    axes[0].barh(top.feature, top.gain_share, color=colors)
    axes[0].set_xlabel("Share of frozen-model tree gain")
    axes[0].set_title("Promoted selector: top objective gain")
    x = [0, 1]
    width = 0.36
    axes[1].bar([value - width / 2 for value in x], control, width, label="preserved final")
    axes[1].bar([value + width / 2 for value in x], promoted, width, label="promoted relationship")
    axes[1].set_xticks(x, ["April gate", "May gate"])
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("Pre-June chronological promotion gates")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(SUBMISSION / "figures" / "05_model_diagnostics.png", dpi=180)
    plt.close(fig)


def build_cover() -> None:
    behavior = json.loads(
        (SUBMISSION / "tables" / "behavior_summary.json").read_text()
    )

    relationship = json.loads(
        (SUBMISSION / "tables" / "target_relationship_validation.json").read_text()
    )
    may_selector = relationship["windows"]["may"]["final_plus_expanded_relationship"]

    curve = json.loads(
        (SUBMISSION / "tables" / "curve_replay_results.json").read_text()
    )
    selective = curve["results"]["offset_118"]["selective_two_stage"]

    pnl_low = selective["fixed_quote"]["fee_0.0095"][
        "p99_capped_total_pnl_sol_supported"
    ]
    pnl_high = selective["fixed_token"]["fee_0.0095"][
        "p99_capped_total_pnl_sol_supported"
    ]

    # 2:1 ratio matches Kaggle's 560x280 card requirement.
    fig = plt.figure(figsize=(12, 6), facecolor="#071923")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(
        0.06, 0.78,
        "SIX SECONDS TO DECIDE",
        fontsize=36,
        weight="bold",
        color="white",
    )
    ax.text(
        0.06, 0.68,
        "Reverse-engineering a Solana zero-block sniper",
        fontsize=20,
        color="#8ED5E6",
    )

    ax.plot(
        [0.06, 0.94],
        [0.61, 0.61],
        color="#E07A2D",
        linewidth=3,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cards = [
        ("15,927", "core bot entries"),
        (f"{behavior['zero_slot']['share']:.1%}", "same-slot entries"),
        (f"{may_selector['pr_auc']:.3f}", "May PR-AUC"),
        (f"{may_selector['precision']:.1%}", "May precision"),
        (f"+{pnl_low:.1f}–{pnl_high:.1f}", "+118 capped P&L (SOL)*"),
    ]

    x_positions = [0.07, 0.245, 0.42, 0.595, 0.77]

    for x, (value, label) in zip(x_positions, cards, strict=True):
        ax.text(
            x, 0.43,
            value,
            fontsize=24,
            weight="bold",
            color="white",
        )
        ax.text(
            x, 0.35,
            label,
            fontsize=11,
            color="#B9CAD1",
        )

    ax.text(
        0.06,
        0.14,
        "Strict-prior features  •  chronological validation  •  latency/slippage falsification",
        fontsize=14,
        color="#8ED5E6",
    )

    ax.text(
        0.06,
        0.07,
        "*Exact standard-curve intent bounds at +118 and 0.95% fees; supported cases only.",
        fontsize=10,
        color="#B9CAD1",
    )

    fig.savefig(
        SUBMISSION / "figures" / "cover.png",
        dpi=180,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

def build_third_pass_summary() -> None:
    historical = json.loads((SUBMISSION / "tables" / "historical_outcome_audit.json").read_text())
    developer_sell = json.loads(
        (SUBMISSION / "tables" / "developer_sell_outcome_results.json").read_text()
    )
    strategy = json.loads((SUBMISSION / "tables" / "profitable_disagreement_results.json").read_text())
    curve = json.loads((SUBMISSION / "tables" / "curve_replay_results.json").read_text())
    backtest = json.loads((SUBMISSION / "tables" / "backtest_results.json").read_text())

    selections = strategy["june_reporting_only"]["selection"]
    immediate = strategy["june_reporting_only"]["backtest"]["immediate"]
    marginal = strategy["june_reporting_only"]["backtest"]["offset_118"]
    name_map = {
        "baseline": "baseline_replica",
        "quality_augmented": "quality_augmented_replica",
        "two_stage": "two_stage",
        "selective_two_stage": "selective_two_stage",
    }
    rows = []
    for label, curve_name in name_map.items():
        selection = selections[label]
        marginal_row = marginal[curve_name]
        fixed_quote = curve["results"]["offset_118"][curve_name]["fixed_quote"]["fee_0.0095"]
        fixed_token = curve["results"]["offset_118"][curve_name]["fixed_token"]["fee_0.0095"]
        rows.append(
            {
                "policy": label,
                "june_entries": selection["entries"],
                "target_overlap": selection["target_overlap"],
                "precision_vs_target": selection["precision_vs_target"],
                "recall_of_target": selection["recall_of_target"],
                "network_adjusted_marginal_immediate_hit_rate": immediate[curve_name]["hit_rate"],
                "network_adjusted_marginal_immediate_median_roi": immediate[curve_name]["median_roi"],
                "network_adjusted_marginal_offset118_fill_rate": marginal_row["fill_rate"],
                "network_adjusted_marginal_offset118_hit_rate": marginal_row["hit_rate"],
                "network_adjusted_marginal_offset118_median_roi": marginal_row["median_roi"],
                "network_adjusted_marginal_offset118_p99_capped_pnl_sol": marginal_row["total_pnl_sol_roi_capped_at_p99"],
                "network_adjusted_marginal_offset118_max_drawdown_sol": marginal_row["max_drawdown_sol"],
                "curve_fixed_quote_coverage": fixed_quote["coverage"],
                "curve_fixed_quote_median_roi": fixed_quote["median_net_roi"],
                "curve_fixed_quote_p99_capped_pnl_sol": fixed_quote["p99_capped_total_pnl_sol_supported"],
                "curve_fixed_token_coverage": fixed_token["coverage"],
                "curve_fixed_token_median_roi": fixed_token["median_net_roi"],
                "curve_fixed_token_p99_capped_pnl_sol": fixed_token["p99_capped_total_pnl_sol_supported"],
                "actual_target_cashflow_fully_fee_adjusted_hit_rate": np.nan,
                "actual_target_cashflow_fully_fee_adjusted_pnl_usd": np.nan,
                "actual_target_cashflow_fully_fee_adjusted_roi": np.nan,
            }
        )
    target_marginal = marginal["target_equal_stake"]
    target_immediate = immediate["target_equal_stake"]
    target_curve = curve["results"]["offset_118"]["target_equal_stake"]
    target_quote = target_curve["fixed_quote"]["fee_0.0095"]
    target_token = target_curve["fixed_token"]["fee_0.0095"]
    target_cashflow = backtest["actual_target_cashflow_june"]
    rows.insert(
        0,
        {
            "policy": "target_equal_stake_counterfactual",
            "june_entries": target_marginal["selected_tokens"],
            "target_overlap": np.nan,
            "precision_vs_target": np.nan,
            "recall_of_target": np.nan,
            "network_adjusted_marginal_immediate_hit_rate": target_immediate["hit_rate"],
            "network_adjusted_marginal_immediate_median_roi": target_immediate["median_roi"],
            "network_adjusted_marginal_offset118_fill_rate": target_marginal["fill_rate"],
            "network_adjusted_marginal_offset118_hit_rate": target_marginal["hit_rate"],
            "network_adjusted_marginal_offset118_median_roi": target_marginal["median_roi"],
            "network_adjusted_marginal_offset118_p99_capped_pnl_sol": target_marginal[
                "total_pnl_sol_roi_capped_at_p99"
            ],
            "network_adjusted_marginal_offset118_max_drawdown_sol": target_marginal[
                "max_drawdown_sol"
            ],
            "curve_fixed_quote_coverage": target_quote["coverage"],
            "curve_fixed_quote_median_roi": target_quote["median_net_roi"],
            "curve_fixed_quote_p99_capped_pnl_sol": target_quote[
                "p99_capped_total_pnl_sol_supported"
            ],
            "curve_fixed_token_coverage": target_token["coverage"],
            "curve_fixed_token_median_roi": target_token["median_net_roi"],
            "curve_fixed_token_p99_capped_pnl_sol": target_token[
                "p99_capped_total_pnl_sol_supported"
            ],
            "actual_target_cashflow_fully_fee_adjusted_hit_rate": target_cashflow[
                "fully_fee_adjusted_hit_rate"
            ],
            "actual_target_cashflow_fully_fee_adjusted_pnl_usd": target_cashflow[
                "fully_fee_adjusted_pnl_usd"
            ],
            "actual_target_cashflow_fully_fee_adjusted_roi": target_cashflow[
                "fully_fee_adjusted_roi_on_buy_plus_costs"
            ],
        },
    )
    pd.DataFrame(rows).to_csv(SUBMISSION / "tables" / "third_pass_head_to_head.csv", index=False)

    windows = developer_sell["windows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    periods = ["April", "May"]
    x = range(2)
    baseline = [windows[p.lower()]["baseline"]["pr_auc"] for p in periods]
    creator_fee = [windows[p.lower()]["creator_fee_quality"]["pr_auc"] for p in periods]
    augmented = [
        windows[p.lower()]["creator_fee_plus_developer_sell"]["pr_auc"]
        for p in periods
    ]
    axes[0].plot(x, baseline, marker="o", linewidth=2.5, label="control")
    axes[0].plot(x, creator_fee, marker="o", linewidth=2.5, label="+ creator fees")
    axes[0].plot(x, augmented, marker="o", linewidth=2.5, label="+ developer sells")
    axes[0].set_xticks(list(x), periods)
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_title("Pre-June chronological validation")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    display_names = ["Control", "Prior quality", "Equal-count", "Selective"]
    low = []
    high = []
    for curve_name in name_map.values():
        result = curve["results"]["offset_118"][curve_name]
        low.append(result["fixed_quote"]["fee_0.0095"]["median_net_roi"])
        high.append(result["fixed_token"]["fee_0.0095"]["median_net_roi"])
    positions = list(range(len(display_names)))
    axes[1].bar(
        [position - 0.19 for position in positions],
        low,
        width=0.38,
        color="#D76A53",
        label="fixed-quote replay",
    )
    axes[1].bar(
        [position + 0.19 for position in positions],
        high,
        width=0.38,
        color="#4C9F70",
        label="fixed-token replay",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Median net ROI")
    axes[1].set_title("+118 exact-curve intent bounds")
    axes[1].set_xticks(positions, display_names, rotation=15)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(SUBMISSION / "figures" / "07_third_pass_summary.png", dpi=180)
    plt.close(fig)


def build_primary_part3_figure() -> None:
    """Plot the aggregate primary Part 3 result without loading row-level outcomes."""
    result = json.loads(
        (SUBMISSION / "tables" / "target_relationship_primary_backtest.json").read_text()
    )
    delays = ["delay_0", "delay_1", "delay_2"]
    labels = ["+0", "+1", "+2"]
    selector = [
        result["marginal_execution"][delay]["reproduced_selector"]
        for delay in delays
    ]
    target = [
        result["marginal_execution"][delay]["target_equal_stake"]
        for delay in delays
    ]
    x = np.arange(len(delays))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        x - width / 2,
        [row["hit_rate"] for row in selector],
        width,
        label="corrected selector",
        color="#176B87",
    )
    axes[0].bar(
        x + width / 2,
        [row["hit_rate"] for row in target],
        width,
        label="target equal-stake",
        color="#E07A2D",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Hit rate")
    axes[0].set_title("Primary marginal backtest")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        x - width / 2,
        [row["median_roi"] for row in selector],
        width,
        label="corrected selector",
        color="#176B87",
    )
    axes[1].bar(
        x + width / 2,
        [row["median_roi"] for row in target],
        width,
        label="target equal-stake",
        color="#E07A2D",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Median network-cost-adjusted ROI")
    axes[1].set_title("Delay sensitivity")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(SUBMISSION / "figures" / "06_backtest_comparison.png", dpi=180)
    plt.close(fig)


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    cells = [
        nbf.v4.new_markdown_cell('# Six Seconds to Decide\n\nEnd-to-end, leakage-safe reconstruction of the Solana sniper. This notebook runs from a normal repository checkout. Competition data is expected in the documented `data/raw/` layout, or through the optional `SOLANA_*` input-path environment overrides defined in `config.py`. Raw competition data is not included in the repository.'),
        nbf.v4.new_markdown_cell('## Decision clock\n\n`t_decision` is the token deployment time. A previous target buy is available only when `target_buy_time < current candidate block_time`; equality, the target reaction to the current token, and every future target action are excluded. Entry features use the signed deployment message and deployer information observable strictly before deployment. Transaction `meta`, trades, candles, and landed Jito data are evaluation-only.'),
        nbf.v4.new_code_cell('from contextlib import redirect_stdout\nfrom io import StringIO\nfrom pathlib import Path\nimport json\nimport os\nimport sys\nimport time\n\nimport pandas as pd\nfrom IPython.display import Image, display\n\ncwd = Path.cwd().resolve()\nif (cwd / "pyproject.toml").exists():\n    ROOT = cwd\nelif (cwd.parent / "pyproject.toml").exists():\n    ROOT = cwd.parent\nelse:\n    raise RuntimeError(\n        f"Could not locate the repository root from {cwd}. "\n        "Run the notebook from the repository root or submission directory."\n    )\n\nos.chdir(ROOT)\nsys.path.insert(0, str(ROOT / "src"))\n\nfrom solana_sniper_reverse_engineering.config import (\n    BOUGHT_ACTIVITY,\n    BOUGHT_INDEX,\n    BOUGHT_TXS,\n    JITO_TRANSACTIONS,\n    JUNE_TRADES,\n    NOT_BOUGHT_ACTIVITY,\n    NOT_BOUGHT_INDEX,\n    NOT_BOUGHT_TXS,\n    TARGET_ACTIVITY,\n    TARGET_TXS,\n    TARGET_TX_INDEX,\n    ensure_output_dirs,\n)\n\nensure_output_dirs()\n\nrequired_inputs = {\n    "SOLANA_BOUGHT_TXS": BOUGHT_TXS,\n    "SOLANA_BOUGHT_INDEX": BOUGHT_INDEX,\n    "SOLANA_BOUGHT_ACTIVITY": BOUGHT_ACTIVITY,\n    "SOLANA_NOT_BOUGHT_TXS": NOT_BOUGHT_TXS,\n    "SOLANA_NOT_BOUGHT_INDEX": NOT_BOUGHT_INDEX,\n    "SOLANA_NOT_BOUGHT_ACTIVITY": NOT_BOUGHT_ACTIVITY,\n    "SOLANA_TARGET_ACTIVITY": TARGET_ACTIVITY,\n    "SOLANA_TARGET_TXS": TARGET_TXS,\n    "SOLANA_TARGET_TX_INDEX": TARGET_TX_INDEX,\n    "SOLANA_JITO_TRANSACTIONS": JITO_TRANSACTIONS,\n    "SOLANA_JUNE_TRADES": JUNE_TRADES,\n}\nmissing_inputs = {\n    name: str(path) for name, path in required_inputs.items() if not Path(path).exists()\n}\nif missing_inputs:\n    raise FileNotFoundError(\n        "Missing required competition inputs. Place them in the documented data/raw layout "\n        "or set the corresponding SOLANA_* environment variables before starting Python:\\n"\n        + json.dumps(missing_inputs, indent=2)\n    )\n\n\ndef run_quietly(function, *args, **kwargs):\n    runner_stdout = StringIO()\n    with redirect_stdout(runner_stdout):\n        return function(*args, **kwargs)\n\n\ndef display_image(filename):\n    display(Image(filename=str(Path(filename))))'),
        nbf.v4.new_markdown_cell("## 1. Target behavior and fee ledger\n\nThe behavior reconstruction measures the target wallet's entry timing, sizing, exits, and realized cash flow. The fee ledger treats `cost_usd` as quote principal, charges inclusive transaction gas once, and applies the separately observed Pump route cost without double-counting priority/tip components."),
        nbf.v4.new_code_cell('from solana_sniper_reverse_engineering.behavior import run as run_behavior\n\nbehavior = run_quietly(run_behavior)\ndisplay(pd.DataFrame([\n    ("core deployment tokens bought", behavior["scope"]["core_bought_deployment_tokens"]),\n    ("wallet bought positions", behavior["scope"]["wallet_bought_tokens"]),\n    ("entry USD mean", behavior["entry_usd_core"]["mean"]),\n    ("entry USD median", behavior["entry_usd_core"]["median"]),\n    ("same-slot share", behavior["zero_slot"]["share"]),\n    ("same-slot median tx delta", behavior["same_slot_position"]["median_tx_delta"]),\n    ("hold median seconds", behavior["hold_seconds"]["median"]),\n    ("partial-exit share", behavior["exit_structure"]["partial_exit_share"]),\n    ("quote-principal buys USD", behavior["cashflow_performance_bought_positions"]["quote_principal_buy_usd"]),\n    ("quote-principal sells USD", behavior["cashflow_performance_bought_positions"]["quote_principal_sell_usd"]),\n    ("inclusive network cost USD", behavior["cashflow_performance_bought_positions"]["network_execution_cost_usd"]),\n    ("separate Pump cost USD", behavior["cashflow_performance_bought_positions"]["pump_separate_cost_usd"]),\n    ("fully fee-adjusted P&L USD", behavior["cashflow_performance_bought_positions"]["fully_fee_adjusted_pnl_usd"]),\n    ("fully fee-adjusted hit rate", behavior["cashflow_performance_bought_positions"]["fully_fee_adjusted_hit_rate"]),\n], columns=["metric", "value"]))'),
        nbf.v4.new_code_cell('display_image("submission/figures/02_entry_latency.png")\ndisplay_image("submission/figures/03_holds_and_exits.png")'),
        nbf.v4.new_markdown_cell('## 2. Leakage-safe feature reconstruction\n\nThe reproduction feature matrix is a local derived artifact, not an input to the published solution. If the exact cache is absent, the source builders stream the competition inputs and rebuild signed-message features, strict-prior deployer state, historical quality state, corrected same-second developer-sell tie groups, and live target–signer relationship state. If the cache already exists, its manifest and population counts are verified before use.'),
        nbf.v4.new_code_cell('from solana_sniper_reverse_engineering.frozen_reproduction import build_reproduction_feature_cache\n\nfeature_path, feature_state = run_quietly(build_reproduction_feature_cache)\nassert feature_state["mode"] in {"rebuilt_from_authorized_inputs", "verified_private_working_cache"}\ndisplay(\n    pd.Series(\n        {key: value for key, value in feature_state.items() if key != "path"},\n        name="value",\n    ).rename_axis("feature-state verification").to_frame()\n)'),
        nbf.v4.new_code_cell('feature_dictionary = pd.read_csv("submission/tables/feature_dictionary.csv")\ndisplay(feature_dictionary[["family", "temporal_construction", "legality"]])'),
        nbf.v4.new_markdown_cell('## 3. Promoted target–signer relationship selector\n\nThe missing policy signal is live memory of deployment signers the target bought from previously. The frozen final recipe fits `2026-03-12 <= block_time < 2026-05-01`; April therefore enters the final weights. Full May selected the promoted family and fixed threshold `0.23211809647507783`, but May labels do not enter the fitted weights. June is reporting only.\n\nApril and May below are the frozen promotion references. June is freshly scored by a newly fitted copy of the immutable recipe.'),
        nbf.v4.new_code_cell('from solana_sniper_reverse_engineering.frozen_reproduction import run_frozen_reproduction\n\nfit_started = time.monotonic()\nreproduction = run_quietly(run_frozen_reproduction, force_backtest=True)\nreproduction_runtime_seconds = time.monotonic() - fit_started\n\nassert reproduction["status"] == "PASS_FRESH_TRAINING_AND_JUNE_SCORING"\nassert reproduction["training"]["rows_after_frozen_sampling"] == 650_194\nassert reproduction["june_predictions"]["generated_by_fresh_model"] is True\n\nrelationship_validation = json.loads(\n    Path("submission/tables/target_relationship_validation.json").read_text()\n)\nclassification_rows = {\n    "April promotion gate": relationship_validation["windows"]["april"]["final_plus_expanded_relationship"],\n    "May promotion / threshold gate": relationship_validation["windows"]["may"]["final_plus_expanded_relationship"],\n    "Fresh June reproduction": reproduction["june_metrics"],\n}\ndisplay(pd.DataFrame(classification_rows).T[\n    ["prevalence", "pr_auc", "precision", "recall", "f1", "predicted_entries", "true_positives"]\n])\n\ndisplay(pd.Series({\n    "frozen threshold": reproduction["june_metrics"]["threshold"],\n    "fresh sampled training rows": reproduction["training"]["rows_after_frozen_sampling"],\n    "fresh June rows": reproduction["june_metrics"]["rows"],\n    "fresh fit + score + backtest seconds": reproduction_runtime_seconds,\n}, name="reproduction summary"))'),
        nbf.v4.new_markdown_cell('## 4. Reverse-engineered rule and feature importance\n\nTree gain is descriptive, correlated importance rather than a causal decomposition. The dominant signal is whether a signer has launched repeatedly since the target last bought from it, backed by recent target-buy rates, deployment spacing, signed developer commitment, prior activity/quality, and compute-budget intent.'),
        nbf.v4.new_code_cell('relationship_importance = pd.read_csv(\n    "submission/tables/target_relationship_feature_importance.csv"\n)\nrelationship_dictionary = pd.read_csv(\n    "submission/tables/target_relationship_feature_dictionary.csv"\n)\ndisplay(relationship_importance.head(10)[["rank", "feature", "gain_share", "feature_family"]])\ndisplay(relationship_dictionary[["family", "temporal_construction", "legality"]])\ndisplay_image("submission/figures/05_model_diagnostics.png")'),
        nbf.v4.new_markdown_cell('## 5. Classification and economic selection are different tasks\n\nThe promoted classifier estimates whether the target wallet will buy a deployment. The separate economic selector estimates a seven-day creator-fee outcome using only labels mature before each evaluation boundary. It is retained only for the secondary exact-execution study below; its weaker target-classification PR-AUC is not presented as an alternative Part 2 result.'),
        nbf.v4.new_markdown_cell('## 6. Primary Part 3: source-built marginal-price backtest\n\nFresh June selections from the promoted classifier feed the primary 0/1/2-slot marginal execution backtest. It uses the same 1.9753 SOL stake and six-second exit, subtracts the inclusive 0.09101 SOL two-leg network cost exactly once, and remains gross of proportional Pump swap fees. Immediate execution is an optimistic bound; whole-slot delays do not claim mempool visibility.'),
        nbf.v4.new_code_cell('reproduction_backtest = reproduction["part3_reproduction"]\n\ndisplay(pd.Series(reproduction_backtest["selection_overlap"], name="June selection overlap"))\n\nbacktest_rows = []\nfor delay in (0, 1, 2):\n    result = reproduction_backtest["marginal_execution"][f"delay_{delay}"]\n    for label, key in (\n        ("Corrected Part 2 selector", "reproduced_selector"),\n        ("Target equal-stake diagnostic", "target_equal_stake"),\n    ):\n        row = result[key]\n        backtest_rows.append({\n            "delay slots": delay,\n            "policy": label,\n            "fill rate": row["fill_rate"],\n            "hit rate": row["hit_rate"],\n            "median ROI": row["median_roi"],\n            "p99-capped P&L SOL": row["total_pnl_sol_roi_capped_at_p99"],\n            "max drawdown SOL": row["max_drawdown_sol"],\n        })\n\ndisplay(pd.DataFrame(backtest_rows))\n\nactual = reproduction_backtest["actual_target_cashflow_june"]\ndisplay(pd.Series({\n    "quote principal paid USD": actual["quote_principal_buy_usd"],\n    "quote principal received USD": actual["quote_principal_sell_usd"],\n    "inclusive network cost USD": actual["network_execution_cost_usd"],\n    "separate Pump cost USD": actual["pump_separate_cost_usd"],\n    "fully fee-adjusted P&L USD": actual["fully_fee_adjusted_pnl_usd"],\n    "fully fee-adjusted hit rate": actual["fully_fee_adjusted_hit_rate"],\n    "ROI on buys + defensible costs": actual["fully_fee_adjusted_roi_on_buy_plus_costs"],\n}, name="Actual target June cash flow"))\n\ndisplay_image("submission/figures/06_backtest_comparison.png")'),
        nbf.v4.new_markdown_cell('## 7. Exact Pump mechanics and bounded conclusions\n\nThe secondary integer constant-product replay refits the frozen control/quality/economic models from source, rebuilds their June memberships, and recomputes curve outcomes from the June Pump trade table. It includes our own curve impact, intervening supported events, 0.95–1.25% proportional fee assumptions, and the same inclusive network cost. Unsupported curves, migrations, delayed intent, slippage limits, and counterfactual completions remain excluded rather than guessed.'),
        nbf.v4.new_code_cell('from solana_sniper_reverse_engineering.frozen_reproduction import reproduce_frozen_exact_replay\n\nexact_reproduction = run_quietly(reproduce_frozen_exact_replay)\nassert exact_reproduction["status"] == "PASS_FRESH_STRATEGY_SELECTION_AND_EXACT_CURVE_REPLAY"\nassert exact_reproduction["aggregate_reference_match"] is True\n\ncurve = json.loads(Path("submission/tables/curve_replay_results.json").read_text())\npolicy_names = {\n    "Preserved control": "baseline_replica",\n    "Prior quality selector": "quality_augmented_replica",\n    "Equal-count two-stage": "two_stage",\n    "Selective two-stage": "selective_two_stage",\n}\nexact_rows = []\nfor label, policy in policy_names.items():\n    result = curve["results"]["offset_118"][policy]\n    quote = result["fixed_quote"]["fee_0.0095"]\n    token = result["fixed_token"]["fee_0.0095"]\n    exact_rows.append({\n        "strategy": label,\n        "fixed-quote coverage": quote["coverage"],\n        "fixed-token coverage": token["coverage"],\n        "fixed-quote median ROI": quote["median_net_roi"],\n        "fixed-token median ROI": token["median_net_roi"],\n        "fixed-quote p99-capped P&L SOL": quote["p99_capped_total_pnl_sol_supported"],\n        "fixed-token p99-capped P&L SOL": token["p99_capped_total_pnl_sol_supported"],\n    })\ndisplay(pd.DataFrame(exact_rows))\n\nselective_125 = curve["results"]["offset_118"]["selective_two_stage"]\ndisplay(pd.Series({\n    "1.25% fixed-quote p99-capped P&L SOL": selective_125["fixed_quote"]["fee_0.0125"]["p99_capped_total_pnl_sol_supported"],\n    "1.25% fixed-token p99-capped P&L SOL": selective_125["fixed_token"]["fee_0.0125"]["p99_capped_total_pnl_sol_supported"],\n    "fresh exact replay seconds": exact_reproduction["elapsed_seconds"],\n}, name="Selective strategy upper-fee bound"))'),
        nbf.v4.new_markdown_cell('## Conclusion\n\nThe dominant reverse-engineered policy signal is strict online memory of deployment signers the target wallet bought from previously. The signal is strong across both pre-June chronological gates: PR-AUC reaches **0.282063 in April** and **0.385999 in May**, with the May promotion gate achieving **41.47% precision** and **64.13% recall**. The frozen model then scores **0.2047103771 PR-AUC in June**, with **29.32% precision**, **42.60% recall**, and **1,787 true positives across 6,094 selections**. June is materially weaker than April and May, but it is reporting-only: no June labels were used for training, threshold selection, or redesign.\n\nThe recovered policy is therefore not a static signer whitelist. It is a live, strict-prior relationship state that updates only after a target buy has actually occurred. The strongest feature, `deployments_since_prior_target_buy`, accounts for **58.59% of model gain**, with recent target-buy rates and ordinary deployer-history features providing additional context.\n\nExecution is the main economic constraint. In the primary marginal-price backtest, immediate execution for the corrected selector has a **66.74% hit rate** and **+20.27% median ROI**, but delaying entry by one slot reduces the hit rate to **34.93%** and the median ROI to **-10.71%**. The target wallet\'s actual June activity remains profitable at **+$185,610.17 fully fee-adjusted P&L** and **16.79% ROI**, but its variable sizing and multi-exit behavior are not directly comparable to the equal-stake replica.\n\nThe stricter exact Pump replay reinforces the latency constraint rather than overturning it. At the observed +118 transaction-position diagnostic, the selective two-stage strategy has **46.86-47.93% supported coverage** and median fully modeled ROI of **-8.46% to -7.06%** under the two observed buy-intent bounds. Its p99-capped P&L ranges from **-9.0 to +38.2 SOL at 0.95% fees**, and from **-17.0 to +29.9 SOL at 1.25% fees**.\n\nThe main result is therefore a reproducible reconstruction of a measurable target-selection policy, not a claim of guaranteed trading profitability. The classification signal generalizes into June, but the economic edge is highly sensitive to execution latency, exact transaction intent, and curve mechanics.'),
    ]
    notebook["cells"] = cells
    nbf.write(notebook, SUBMISSION / "final_notebook.ipynb")

def main() -> None:
    build_public_reproduction_recipe()
    build_promoted_selector_explainability()
    build_third_pass_summary()
    build_primary_part3_figure()
    build_cover()
    build_notebook()
    print("built submission summary figure/table, cover, and final notebook")


if __name__ == "__main__":
    main()