#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import nbformat as nbf
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"


def build_final_selector_explainability() -> None:
    """Extract presentation tables from the frozen promoted selector.

    This performs no fitting, threshold selection, or policy change. It applies the
    already-promoted model to the unchanged May validation population using the same
    deterministic permutation/effect routines as the preserved-control report.
    """
    from solana_sniper_reverse_engineering.modeling import (
        ModelBundle,
        _feature_effects,
        _permutation_importance,
    )
    from solana_sniper_reverse_engineering.third_pass import (
        JUNE_START,
        MAY_START,
        _joined_frame,
    )

    model_dir = ROOT / "artifacts" / "models" / "quality_augmented_final"
    metadata = json.loads((model_dir / "model_features.json").read_text())
    bundle = ModelBundle(
        model=joblib.load(model_dir / "final_model.joblib"),
        numeric_features=metadata["numeric_features"],
        categorical_features=metadata["categorical_features"],
        feature_names_out=metadata["transformed_feature_names"],
        name="historical_outcome_augmented_active_era_lightgbm",
    )
    validation = _joined_frame(
        f"f.block_time>={MAY_START} AND f.block_time<{JUNE_START}",
        bundle.numeric_features,
        bundle.categorical_features,
        quality=True,
        dev_sell=True,
    )
    importance = _permutation_importance(
        bundle, validation, np.random.default_rng(20260811)
    ).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance.to_csv(
        SUBMISSION / "tables" / "final_selector_feature_importance.csv", index=False
    )
    effects = _feature_effects(bundle, validation, importance)
    effects.to_csv(
        SUBMISSION / "tables" / "final_selector_feature_effects.csv", index=False
    )


def build_cover() -> None:
    behavior = json.loads((SUBMISSION / "tables" / "behavior_summary.json").read_text())
    final_selector = json.loads(
        (SUBMISSION / "tables" / "developer_sell_outcome_results.json").read_text()
    )["june_reporting_only"]["creator_fee_plus_developer_sell"]
    curve = json.loads((SUBMISSION / "tables" / "curve_replay_results.json").read_text())
    selective = curve["results"]["offset_118"]["selective_two_stage"]
    best_median = max(
        selective[intent]["fee_0.0095"]["median_net_roi"]
        for intent in ("fixed_quote", "fixed_token")
    )
    fig = plt.figure(figsize=(12, 6.3), facecolor="#071923")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.06, 0.78, "SIX SECONDS TO DECIDE", fontsize=36, weight="bold", color="white")
    ax.text(0.06, 0.68, "Reverse-engineering a Solana zero-block sniper", fontsize=20, color="#8ED5E6")
    ax.plot([0.06, 0.94], [0.61, 0.61], color="#E07A2D", linewidth=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cards = [
        ("15,927", "core bot entries"),
        (f"{behavior['zero_slot']['share']:.1%}", "same-slot entries"),
        (f"{final_selector['pr_auc']:.3f}", "final June PR-AUC"),
        (f"{final_selector['precision']:.1%}", "final June precision"),
        (f"{best_median:.1%}", "+118 median ROI*"),
    ]
    x_positions = [0.07, 0.245, 0.42, 0.595, 0.77]
    for x, (value, label) in zip(x_positions, cards, strict=True):
        ax.text(x, 0.43, value, fontsize=24, weight="bold", color="white")
        ax.text(x, 0.35, label, fontsize=11, color="#B9CAD1")
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
        "*Best exact standard-curve intent bound at 0.95% fees; supported cases only.",
        fontsize=10,
        color="#B9CAD1",
    )
    fig.savefig(SUBMISSION / "figures" / "cover.png", dpi=180, facecolor=fig.get_facecolor())
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
                "marginal_immediate_hit_rate": immediate[curve_name]["hit_rate"],
                "marginal_immediate_median_roi": immediate[curve_name]["median_roi"],
                "marginal_offset118_fill_rate": marginal_row["fill_rate"],
                "marginal_offset118_hit_rate": marginal_row["hit_rate"],
                "marginal_offset118_median_roi": marginal_row["median_roi"],
                "marginal_offset118_p99_capped_pnl_sol": marginal_row["total_pnl_sol_roi_capped_at_p99"],
                "marginal_offset118_max_drawdown_sol": marginal_row["max_drawdown_sol"],
                "curve_fixed_quote_coverage": fixed_quote["coverage"],
                "curve_fixed_quote_median_roi": fixed_quote["median_net_roi"],
                "curve_fixed_quote_p99_capped_pnl_sol": fixed_quote["p99_capped_total_pnl_sol_supported"],
                "curve_fixed_token_coverage": fixed_token["coverage"],
                "curve_fixed_token_median_roi": fixed_token["median_net_roi"],
                "curve_fixed_token_p99_capped_pnl_sol": fixed_token["p99_capped_total_pnl_sol_supported"],
                "actual_target_cashflow_hit_rate": np.nan,
                "actual_target_cashflow_net_pnl_usd": np.nan,
                "actual_target_cashflow_net_roi": np.nan,
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
            "marginal_immediate_hit_rate": target_immediate["hit_rate"],
            "marginal_immediate_median_roi": target_immediate["median_roi"],
            "marginal_offset118_fill_rate": target_marginal["fill_rate"],
            "marginal_offset118_hit_rate": target_marginal["hit_rate"],
            "marginal_offset118_median_roi": target_marginal["median_roi"],
            "marginal_offset118_p99_capped_pnl_sol": target_marginal[
                "total_pnl_sol_roi_capped_at_p99"
            ],
            "marginal_offset118_max_drawdown_sol": target_marginal[
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
            "actual_target_cashflow_hit_rate": target_cashflow[
                "hit_rate_net_cashflow"
            ],
            "actual_target_cashflow_net_pnl_usd": target_cashflow["net_pnl_usd"],
            "actual_target_cashflow_net_roi": target_cashflow[
                "net_roi_on_buy_plus_fees"
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

    display_names = ["Control", "Final imitation", "Equal-count", "Selective"]
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


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# Six Seconds to Decide\n\n"
            "End-to-end, leakage-safe reconstruction of the Solana sniper. This notebook is a clean orchestration layer over the tested repository modules; detailed outputs are cached outside Git."
        ),
        nbf.v4.new_markdown_cell(
            "## Decision clock\n\n"
            "`t_decision` is the moment of deployment. Entry features are limited to the signed deployment message and deployer activity with `timestamp < deployment.blockTime`. Same-second history, transaction `meta`, target reactions, trades, candles, outcomes, and landed Jito data are excluded. Trades/Jito are introduced only in behavior and backtest sections."
        ),
        nbf.v4.new_code_cell(
            "from contextlib import redirect_stdout\n"
            "from io import StringIO\n"
            "from pathlib import Path\n"
            "import json\n"
            "import os\n"
            "import sys\n"
            "import warnings\n\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "cwd = Path.cwd().resolve()\n"
            "if (cwd / 'pyproject.toml').exists():\n"
            "    ROOT = cwd\n"
            "elif (cwd.parent / 'pyproject.toml').exists():\n"
            "    ROOT = cwd.parent\n"
            "else:\n"
            "    raise RuntimeError(\n"
            "        'Could not locate the repository root. Expected pyproject.toml '\n"
            "        'in the current directory or its parent.'\n"
            "    )\n"
            "os.chdir(ROOT)\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from solana_sniper_reverse_engineering.config import ensure_output_dirs\n"
            "ensure_output_dirs()\n\n"
            "def run_quietly(function, *args, **kwargs):\n"
            "    # Successful runners are verbose; exceptions still propagate and fail the cell.\n"
            "    runner_stdout = StringIO()\n"
            "    with redirect_stdout(runner_stdout):\n"
            "        return function(*args, **kwargs)\n\n"
            "def display_image(filename):\n"
            "    display(Image(filename=str(Path(filename))))"
        ),
        nbf.v4.new_markdown_cell("## 1. Reconstruct target behavior"),
        nbf.v4.new_code_cell(
            "from solana_sniper_reverse_engineering.behavior import run as run_behavior\n"
            "behavior = run_quietly(run_behavior)\n"
            "display(pd.DataFrame([\n"
            "    ('core deployment tokens bought', behavior['scope']['core_bought_deployment_tokens']),\n"
            "    ('wallet bought positions', behavior['scope']['wallet_bought_tokens']),\n"
            "    ('entry USD mean', behavior['entry_usd_core']['mean']),\n"
            "    ('entry USD median', behavior['entry_usd_core']['median']),\n"
            "    ('entry USD standard deviation', behavior['entry_usd_core']['std']),\n"
            "    ('same-slot share', behavior['zero_slot']['share']),\n"
            "    ('same-slot median tx delta', behavior['same_slot_position']['median_tx_delta']),\n"
            "    ('hold median seconds', behavior['hold_seconds']['median']),\n"
            "    ('partial-exit share', behavior['exit_structure']['partial_exit_share']),\n"
            "    ('sell transactions', behavior['exit_structure']['sell_transactions_for_bought_positions']),\n"
            "    ('net cash flow USD', behavior['cashflow_performance_bought_positions']['net_pnl_usd']),\n"
            "], columns=['metric','value']))"
        ),
        nbf.v4.new_code_cell(
            "display_image('submission/figures/02_entry_latency.png')\n"
            "display_image('submission/figures/03_holds_and_exits.png')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Build the legal feature store\n\n"
            "On a fresh run the next cell streams both gzip archives once. It decodes signed message fields only; the 15 GiB negative archive is never expanded in place. Deterministic Parquet caches make reruns fast."
        ),
        nbf.v4.new_code_cell(
            "from solana_sniper_reverse_engineering.message_features import run as extract_messages\n"
            "from solana_sniper_reverse_engineering.feature_store import run as build_features, audit\n"
            "run_quietly(extract_messages)\n"
            "feature_path = run_quietly(build_features)\n"
            "feature_audit = run_quietly(audit, feature_path)\n"
            "display(pd.Series(feature_audit, name='value').rename_axis('audit gate').to_frame())"
        ),
        nbf.v4.new_code_cell(
            "feature_dictionary = pd.read_csv('submission/tables/feature_dictionary.csv')\n"
            "display(feature_dictionary[['family','temporal_construction','legality']])"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Chronological model and interpretation\n\n"
            "The target-active era (March 12–April) trains, May selects the family and fixed threshold, and June is a reporting test. An initial HGB run had already inspected June before LightGBM was introduced, so June is not described as literally untouched. Accuracy is intentionally omitted."
        ),
        nbf.v4.new_code_cell(
            "from solana_sniper_reverse_engineering.modeling import run as run_modeling\n"
            "with warnings.catch_warnings():\n"
            "    warnings.filterwarnings(\n"
            "        'ignore', message=\"'n_jobs' has no effect.*\", category=FutureWarning\n"
            "    )\n"
            "    classification = run_quietly(run_modeling, feature_path)\n"
            "display(pd.DataFrame(classification['experiments_may']).T[\n"
            "    ['pr_auc','precision','recall','f1']\n"
            "])"
        ),
        nbf.v4.new_code_cell(
            "final_importance = pd.read_csv('submission/tables/final_selector_feature_importance.csv')\n"
            "final_effects = pd.read_csv('submission/tables/final_selector_feature_effects.csv')\n"
            "top10_notebook = final_importance.head(10)[['rank','feature','importance_pr_auc_drop']].merge(\n"
            "    final_effects[['feature','direction']].drop_duplicates(), on='feature', how='left'\n"
            ")\n"
            "display(top10_notebook)\n"
            "display_image('submission/figures/05_model_diagnostics.png')"
        ),
        nbf.v4.new_markdown_cell("## 4. Frozen replica, backtest, and head-to-head"),
        nbf.v4.new_code_cell(
            "from solana_sniper_reverse_engineering.backtest import run as run_backtest\n"
            "backtest = run_quietly(run_backtest, force=True)\n"
            "display(backtest['selection_overlap'])\n"
            "display(pd.DataFrame(backtest['primary_fee_results']).T)"
        ),
        nbf.v4.new_code_cell("display_image('submission/figures/06_backtest_comparison.png')"),
        nbf.v4.new_markdown_cell(
            "## 5. Falsification\n\n"
            "The robustness run retrains on a through-March window, validates April, tests top-feature stability, missing-history/dev-buy subgroups, concentration, calibration, and threshold sensitivity. Fee, 0/1/2-slot delay, slippage, tail-cap, and no-fill stresses are part of the backtest."
        ),
        nbf.v4.new_code_cell(
            "from solana_sniper_reverse_engineering.robustness import run as run_robustness\n"
            "robustness = run_quietly(run_robustness)\n"
            "display(robustness['alternate_temporal_validation'])\n"
            "display(robustness['top10_stability'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Targeted methodological audit\n\n"
            "This controlled pass compares January–April with March 12–April training, quantifies redundancy among the top activity signals, and simulates +1 through +250 landed transaction-position offsets plus a pre-June empirical same-slot policy. Trade ordering remains backtest-only."
        ),
        nbf.v4.new_code_cell(
            "from solana_sniper_reverse_engineering.methodological_audit import run as run_audit\n"
            "audit_results = run_quietly(run_audit, feature_path)\n"
            "display({\n"
            "    'active-period decision': audit_results['active_period_training']['decision'],\n"
            "    'May activity-signal median Spearman': audit_results['signal_redundancy']['all_may_pairwise_spearman'],\n"
            "})"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Third-pass research\n\n"
            "The control is preserved. New families are selected only from April/May chronological evidence. Creator-fee claims and a deployer's own sells of prior launches are timestamped outcome proxies—not token ROI—and enter only after observation. The reproducible runner is `.venv/bin/python scripts/run_third_pass.py all`; the cells below present its tracked small outputs."
        ),
        nbf.v4.new_code_cell(
            "historical = json.loads(Path('submission/tables/historical_outcome_audit.json').read_text())\n"
            "developer_sell = json.loads(Path('submission/tables/developer_sell_outcome_results.json').read_text())\n"
            "ranking = json.loads(Path('submission/tables/ranking_hard_negative_results.json').read_text())\n"
            "strategy = json.loads(Path('submission/tables/profitable_disagreement_results.json').read_text())\n"
            "display(pd.DataFrame({\n"
            "    period: {\n"
            "        'control_pr_auc': result['baseline']['pr_auc'],\n"
            "        'creator_fee_pr_auc': result['creator_fee_quality']['pr_auc'],\n"
            "        'all_outcomes_pr_auc': result['creator_fee_plus_developer_sell']['pr_auc'],\n"
            "        'all_minus_control': result['creator_fee_plus_developer_sell']['pr_auc'] - result['baseline']['pr_auc'],\n"
            "    } for period, result in developer_sell['windows'].items()\n"
            "}).T)"
        ),
        nbf.v4.new_code_cell(
            "final_classification = pd.DataFrame({\n"
            "    'May selection': developer_sell['windows']['may']['creator_fee_plus_developer_sell'],\n"
            "    'June reporting': developer_sell['june_reporting_only']['creator_fee_plus_developer_sell'],\n"
            "}).T\n"
            "display(final_classification[\n"
            "    ['prevalence','pr_auc','pr_auc_lift_over_prevalence',\n"
            "     'precision','precision_lift_over_prevalence','recall','f1','predicted_entries']\n"
            "])"
        ),
        nbf.v4.new_code_cell(
            "display(ranking['pre_june_decisions'])\n"
            "display(strategy['pre_june_decision'])\n"
            "display(strategy['pre_june_strategy_operating_point'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Marginal +118 head-to-head\n\n"
            "This table is the common equal-stake **marginal-price diagnostic**, not exact curve replay. Immediate execution is optimistic; +118 is landed transaction ordering rather than wall-clock latency or mempool visibility. Because increasing offsets change the fill population, conditional ROI need not move monotonically."
        ),
        nbf.v4.new_code_cell(
            "head_to_head = pd.read_csv('submission/tables/third_pass_head_to_head.csv')\n"
            "display(head_to_head[[\n"
            "    'policy','june_entries','target_overlap','precision_vs_target',\n"
            "    'recall_of_target','marginal_offset118_fill_rate',\n"
            "    'marginal_offset118_hit_rate','marginal_offset118_median_roi',\n"
            "    'marginal_offset118_p99_capped_pnl_sol',\n"
            "    'marginal_offset118_max_drawdown_sol',\n"
            "]])"
        ),
        nbf.v4.new_markdown_cell(
            "## 9. Exact Pump curve execution bounds\n\n"
            "The tested integer Pump replay inserts our buy and sell, price impact, fees, and intervening events into supported standard curves. The existing runner is invoked below and checked against its tracked JSON result. Fixed-quote and fixed-token continuation bound the missing downstream instruction intent. Nonstandard curves, migrations, counterfactual completions, and possible slippage-limit failures are excluded, not guessed."
        ),
        nbf.v4.new_code_cell(
            "from solana_sniper_reverse_engineering.curve_replay import run_curve_replay\n\n"
            "curve_replay = run_quietly(run_curve_replay, force=False)\n"
            "tracked_curve_replay = json.loads(\n"
            "    Path('submission/tables/curve_replay_results.json').read_text()\n"
            ")\n"
            "assert curve_replay == tracked_curve_replay, 'Curve replay differs from tracked output'\n"
            "curve_coverage_status = pd.read_csv(\n"
            "    'submission/tables/curve_replay_coverage.csv'\n"
            ")\n"
            "assert not curve_coverage_status.empty\n\n"
            "policy_labels = {\n"
            "    'target_equal_stake': 'Target equal-stake diagnostic',\n"
            "    'baseline_replica': 'Preserved control',\n"
            "    'quality_augmented_replica': 'Final imitation',\n"
            "    'two_stage': 'Equal-count two-stage',\n"
            "    'selective_two_stage': 'Selective two-stage',\n"
            "}\n"
            "exact_rows = []\n"
            "for policy, label in policy_labels.items():\n"
            "    for intent in ('fixed_quote', 'fixed_token'):\n"
            "        for fee_key, fee_label in (('fee_0.0095', '0.95%'), ('fee_0.0125', '1.25%')):\n"
            "            metrics = curve_replay['results']['offset_118'][policy][intent][fee_key]\n"
            "            exact_rows.append({\n"
            "                'policy': label,\n"
            "                'landed offset': '+118',\n"
            "                'intent bound': intent.replace('_', '-'),\n"
            "                'swap fee': fee_label,\n"
            "                'selected': metrics['selected_tokens'],\n"
            "                'supported': metrics['supported_tokens'],\n"
            "                'supported coverage': metrics['coverage'],\n"
            "                'median net ROI': metrics['median_net_roi'],\n"
            "                'p99-capped P&L (SOL)': metrics['p99_capped_total_pnl_sol_supported'],\n"
            "            })\n"
            "exact_curve_table = pd.DataFrame(exact_rows)\n"
            "exact_curve_display = exact_curve_table.copy()\n"
            "exact_curve_display['supported coverage'] = exact_curve_display['supported coverage'].map('{:.2%}'.format)\n"
            "exact_curve_display['median net ROI'] = exact_curve_display['median net ROI'].map('{:.2%}'.format)\n"
            "exact_curve_display['p99-capped P&L (SOL)'] = exact_curve_display['p99-capped P&L (SOL)'].map('{:+.1f}'.format)\n"
            "display(exact_curve_display)"
        ),
        nbf.v4.new_code_cell("display_image('submission/figures/07_third_pass_summary.png')"),
        nbf.v4.new_markdown_cell(
            "## Conclusion\n\n"
            "The strongest stable policy signals are deploy spacing, a meaningful dev buy, a latent wallet-scale/sophistication factor, avoidance of industrial deployment cadence, and observed quality of prior launches. Creator-fee and developer-sell history improve both pre-June windows; LambdaRank, hard-negative weighting, metadata expansion, and a missing-history mixture fail. A separately labeled economic ranker raises seven-day creator-fee hit rate in both validation months. Exact-curve +118 results remain negative at the median; the selective strategy has positive capped P&L under both observed buy-intent bounds. See `submission/writeup.md` for bounded claims and limitations."
        ),
    ]
    notebook["cells"] = cells
    nbf.write(notebook, SUBMISSION / "final_notebook.ipynb")


def main() -> None:
    build_final_selector_explainability()
    build_third_pass_summary()
    build_cover()
    build_notebook()
    print("built submission summary figure/table, cover, and final notebook")


if __name__ == "__main__":
    main()
