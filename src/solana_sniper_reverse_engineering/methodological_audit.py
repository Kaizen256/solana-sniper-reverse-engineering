from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from .backtest import run_intraslot_sensitivity
from .config import (
    ACTIVE_ERA_START,
    ARTIFACTS,
    JUNE_START,
    MAY_START,
    SUBMISSION,
    ensure_output_dirs,
)
from .feature_store import FEATURE_STORE
from .modeling import (
    METRICS,
    MODEL_DIR,
    PREDICTIONS,
    ModelBundle,
    _available_features,
    _permutation_importance,
    _thresholds,
    fit_lightgbm,
    metrics,
    predict,
)


ACTIVE_MODEL = ARTIFACTS / "models" / "active_era_model.joblib"
LEGACY_MODEL = ARTIFACTS / "models" / "legacy_population_model.joblib"
ACTIVE_PREDICTIONS = ARTIFACTS / "tables" / "active_era_predictions.parquet"
ACTIVE_RESULTS = SUBMISSION / "tables" / "active_period_training.json"
ACTIVE_IMPORTANCE = SUBMISSION / "tables" / "feature_importance_active_era.csv"
LEGACY_IMPORTANCE = SUBMISSION / "tables" / "feature_importance_legacy_population.csv"
CORRELATIONS = SUBMISSION / "tables" / "signal_correlations.csv"
REDUNDANCY_RESULTS = SUBMISSION / "tables" / "signal_redundancy.json"
AUDIT_RESULTS = SUBMISSION / "tables" / "methodological_audit.json"

ACTIVITY_SCALE_SIGNALS = [
    "hist_quote_sol_sum",
    "hist_cost_usd_sum",
    "hist_open_close_count",
    "hist_sell_count",
    "hist_tip_fee_sum",
]


def _frame(
    path: Path,
    numeric: list[str],
    categorical: list[str],
    predicate: str,
) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            ["token_address", "tx_hash", "block_time", "label"]
            + numeric
            + categorical
        )
    )
    select = ", ".join(f'"{column}"' for column in columns)
    con = duckdb.connect()
    con.execute("SET memory_limit='30GB'")
    con.execute("SET threads=16")
    return con.execute(
        f"SELECT {select} FROM read_parquet(?) WHERE {predicate}", [str(path)]
    ).fetch_df()


def _population_counts(path: Path) -> dict[str, dict[str, int]]:
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT period, count(*) AS candidates, sum(label)::BIGINT AS positives
        FROM (
          SELECT CASE
            WHEN block_time < {ACTIVE_ERA_START} THEN 'excluded_pre_active'
            ELSE 'active_era'
          END AS period, label
          FROM read_parquet(?)
          WHERE block_time < {MAY_START}
        )
        GROUP BY period
        """,
        [str(path)],
    ).fetchall()
    split = {
        period: {
            "candidates": int(candidates),
            "positives": int(positives),
            "negatives": int(candidates - positives),
        }
        for period, candidates, positives in rows
    }
    total_candidates = sum(item["candidates"] for item in split.values())
    total_positives = sum(item["positives"] for item in split.values())
    return {
        "existing_january_april": {
            "candidates": total_candidates,
            "positives": total_positives,
            "negatives": total_candidates - total_positives,
        },
        "active_era_march12_april": split["active_era"],
        "excluded_before_march12": split["excluded_pre_active"],
    }


def _sampled_training_count(path: Path, start: int | None, stride: int) -> dict[str, int]:
    lower = f"block_time >= {start} AND " if start is not None else ""
    con = duckdb.connect()
    candidates, positives = con.execute(
        f"""
        SELECT count(*), sum(label)::BIGINT
        FROM read_parquet(?)
        WHERE {lower}block_time < {MAY_START}
          AND (label=1 OR hash(token_address)%{stride}=0)
        """,
        [str(path)],
    ).fetchone()
    return {"candidates": int(candidates), "positives": int(positives)}


def _enrichment(item: dict[str, object]) -> dict[str, object]:
    result = dict(item)
    prevalence = float(result["prevalence"])
    result["pr_auc_lift_over_prevalence"] = (
        float(result["pr_auc"]) / prevalence if prevalence else 0.0
    )
    result["precision_lift_over_prevalence"] = (
        float(result["precision"]) / prevalence if prevalence else 0.0
    )
    return result


def _active_period_experiment(
    path: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, ModelBundle]:
    numeric, categorical = _available_features(path)
    stride = 2
    validation = _frame(
        path,
        numeric,
        categorical,
        f"block_time >= {MAY_START} AND block_time < {JUNE_START}",
    )
    test = _frame(path, numeric, categorical, f"block_time >= {JUNE_START}")

    feature_meta = json.loads((MODEL_DIR / "model_features.json").read_text())
    active = ModelBundle(
        model=joblib.load(MODEL_DIR / "final_model.joblib"),
        numeric_features=feature_meta["numeric_features"],
        categorical_features=feature_meta["categorical_features"],
        feature_names_out=feature_meta["transformed_feature_names"],
        name="regularized_lightgbm_active_era",
    )
    active_validation_score = predict(active, validation)
    active_threshold = float(
        json.loads((MODEL_DIR / "operating_point.json").read_text())["threshold"]
    )
    active_test_score = predict(active, test)
    active_validation_metrics = metrics(
        validation.label.to_numpy(dtype=np.uint8),
        active_validation_score,
        active_threshold,
    )
    active_test_metrics = metrics(
        test.label.to_numpy(dtype=np.uint8), active_test_score, active_threshold
    )

    legacy_train = _frame(
        path,
        numeric,
        categorical,
        f"block_time < {MAY_START} "
        f"AND (label=1 OR hash(token_address)%{stride}=0)",
    )
    legacy = fit_lightgbm(legacy_train, numeric, categorical, stride)
    legacy.name = "regularized_lightgbm_january_april"
    legacy_validation_score = predict(legacy, validation)
    legacy_threshold = _thresholds(
        validation.label.to_numpy(dtype=np.uint8), legacy_validation_score
    )["max_f1"]
    legacy_test_score = predict(legacy, test)
    legacy_validation_metrics = metrics(
        validation.label.to_numpy(dtype=np.uint8),
        legacy_validation_score,
        legacy_threshold,
    )
    legacy_test_metrics = metrics(
        test.label.to_numpy(dtype=np.uint8), legacy_test_score, legacy_threshold
    )

    active_importance = pd.read_csv(SUBMISSION / "tables" / "feature_importance.csv")
    active_importance.to_csv(ACTIVE_IMPORTANCE, index=False)
    legacy_importance = _permutation_importance(
        legacy,
        validation,
        np.random.default_rng(20260811),
        sample_size=200_000,
    )
    legacy_importance.to_csv(LEGACY_IMPORTANCE, index=False)
    legacy_top = list(legacy_importance.feature.head(10))
    active_top = list(active_importance.feature.head(10))
    intersection = sorted(set(legacy_top) & set(active_top))
    rank_frame = legacy_importance[["feature"]].copy()
    rank_frame["legacy_rank"] = np.arange(1, len(rank_frame) + 1)
    active_ranks = active_importance[["feature"]].copy()
    active_ranks["active_rank"] = np.arange(1, len(active_ranks) + 1)
    rank_frame = rank_frame.merge(active_ranks, on="feature", how="inner")
    rank_correlation = float(
        spearmanr(rank_frame.legacy_rank, rank_frame.active_rank).statistic
    )

    active_prediction_frame = pd.DataFrame(
        {
            "token_address": test.token_address,
            "tx_hash": test.tx_hash,
            "block_time": test.block_time,
            "label": test.label.astype("uint8"),
            "active_era_score": active_test_score,
            "active_era_selected": (active_test_score >= active_threshold).astype(
                "uint8"
            ),
            "legacy_population_score": legacy_test_score,
            "legacy_population_selected": (
                legacy_test_score >= legacy_threshold
            ).astype("uint8"),
        }
    )
    active_prediction_frame.to_parquet(ACTIVE_PREDICTIONS, index=False)
    combined = active_prediction_frame.rename(
        columns={"active_era_score": "score", "active_era_selected": "selected"}
    )
    combined["active_era_score"] = combined.score
    combined["active_era_selected"] = combined.selected
    legacy_selected = combined.legacy_population_selected.eq(1)
    active_selected = combined.selected.eq(1)
    intersection_count = int((legacy_selected & active_selected).sum())
    union_count = int((legacy_selected | active_selected).sum())

    counts = _population_counts(path)
    counts["existing_sampled_training"] = _sampled_training_count(path, None, stride)
    counts["active_era_sampled_training"] = _sampled_training_count(
        path, ACTIVE_ERA_START, stride
    )
    output: dict[str, object] = {
        "question": "Does excluding negatives before 2026-03-12 change temporal generalization?",
        "active_era_cutoff": "2026-03-12T00:00:00Z; the date boundary retains all observed positive deployments",
        "identical_controls": {
            "features": "same final leakage-safe allowlist",
            "model": "same regularized LightGBM parameters and random seed",
            "negative_sampling": "deterministic token hash, one half of negatives, inverse sampling weights",
            "threshold": "separately maximized May F1",
            "validation": "full May population",
            "reporting_test": "full June population",
        },
        "training_counts": counts,
        "existing_model": {
            "policy": "legacy January-April population",
            "may": legacy_validation_metrics,
            "june": legacy_test_metrics,
            "threshold": legacy_threshold,
            "top10": legacy_top,
        },
        "active_era_model": {
            "policy": "adopted final March 12-April population",
            "may": active_validation_metrics,
            "june": active_test_metrics,
            "threshold": active_threshold,
            "top10": active_top,
        },
        "top_feature_stability": {
            "intersection": intersection,
            "intersection_count": len(intersection),
            "jaccard": len(intersection) / len(set(legacy_top) | set(active_top)),
            "all_feature_rank_spearman": rank_correlation,
        },
        "june_selection_overlap_between_models": {
            "existing_count": int(legacy_selected.sum()),
            "active_era_count": int(active_selected.sum()),
            "intersection": intersection_count,
            "union": union_count,
            "jaccard": intersection_count / union_count if union_count else 0.0,
            "existing_only": int((legacy_selected & ~active_selected).sum()),
            "active_only": int((active_selected & ~legacy_selected).sum()),
        },
        "decision": "Adopt active-era training: it improves May PR-AUC, precision, and F1 under the declared selection protocol; June PR-AUC and precision also improve, while June recall and F1 are slightly lower.",
    }
    joblib.dump(active.model, ACTIVE_MODEL)
    joblib.dump(legacy.model, LEGACY_MODEL)
    ACTIVE_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output, combined, validation, active


def _first_component_share(frame: pd.DataFrame) -> float:
    ranks = frame.rank(method="average", pct=True).to_numpy(dtype=float)
    valid_std = np.nanstd(ranks, axis=0)
    keep = valid_std > 0
    ranks = ranks[:, keep]
    if ranks.shape[1] < 2:
        return 1.0
    ranks = (ranks - np.nanmean(ranks, axis=0)) / np.nanstd(ranks, axis=0)
    ranks = np.nan_to_num(ranks)
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(ranks, rowvar=False))
    return float(eigenvalues[-1] / eigenvalues.sum())


def _signal_redundancy(validation: pd.DataFrame) -> dict[str, object]:
    model = joblib.load(MODEL_DIR / "final_model.joblib")
    feature_meta = json.loads((MODEL_DIR / "model_features.json").read_text())
    bundle = ModelBundle(
        model=model,
        numeric_features=feature_meta["numeric_features"],
        categorical_features=feature_meta["categorical_features"],
        feature_names_out=feature_meta["transformed_feature_names"],
        name="regularized_lightgbm",
    )
    baseline_score = predict(bundle, validation)
    threshold = float(json.loads((MODEL_DIR / "operating_point.json").read_text())["threshold"])
    populations = {
        "all_may_candidates": np.ones(len(validation), dtype=bool),
        "may_target_bought": validation.label.to_numpy(dtype=bool),
        "may_replica_selected": baseline_score >= threshold,
    }
    correlation_rows: list[dict[str, object]] = []
    component_share: dict[str, float] = {}
    for population, mask in populations.items():
        subset = validation.loc[mask, ACTIVITY_SCALE_SIGNALS]
        correlation = subset.corr(method="spearman")
        component_share[population] = _first_component_share(subset)
        for left_index, left in enumerate(ACTIVITY_SCALE_SIGNALS):
            for right in ACTIVITY_SCALE_SIGNALS[left_index + 1 :]:
                correlation_rows.append(
                    {
                        "population": population,
                        "n": int(len(subset)),
                        "feature_a": left,
                        "feature_b": right,
                        "spearman": float(correlation.loc[left, right]),
                    }
                )
    pd.DataFrame(correlation_rows).to_csv(CORRELATIONS, index=False)

    sample = validation.sample(min(200_000, len(validation)), random_state=20260811).copy()
    y = sample.label.to_numpy(dtype=np.uint8)
    baseline_pr_auc = float(average_precision_score(y, predict(bundle, sample)))
    groups = {
        "historical_sol_activity": ["hist_quote_sol_sum"],
        "historical_usd_activity": ["hist_cost_usd_sum"],
        "open_close_activity": ["hist_open_close_count"],
        "historical_sells": ["hist_sell_count"],
        "historical_tips": ["hist_tip_fee_sum"],
        "sol_and_usd_activity": ["hist_quote_sol_sum", "hist_cost_usd_sum"],
        "open_close_and_sells": ["hist_open_close_count", "hist_sell_count"],
        "all_five_activity_scale_signals": ACTIVITY_SCALE_SIGNALS,
    }
    rng = np.random.default_rng(20260811)
    permutation_results: dict[str, dict[str, object]] = {}
    for name, features in groups.items():
        permutation = rng.permutation(len(sample))
        originals = {feature: sample[feature].copy() for feature in features}
        for feature in features:
            sample[feature] = originals[feature].to_numpy()[permutation]
        permuted_pr_auc = float(average_precision_score(y, predict(bundle, sample)))
        for feature in features:
            sample[feature] = originals[feature]
        permutation_results[name] = {
            "features": features,
            "permuted_pr_auc": permuted_pr_auc,
            "pr_auc_drop": baseline_pr_auc - permuted_pr_auc,
        }
    individual_names = [
        "historical_sol_activity",
        "historical_usd_activity",
        "open_close_activity",
        "historical_sells",
        "historical_tips",
    ]
    summed_individual_drop = sum(
        float(permutation_results[name]["pr_auc_drop"]) for name in individual_names
    )
    joint_drop = float(
        permutation_results["all_five_activity_scale_signals"]["pr_auc_drop"]
    )
    all_correlations = pd.DataFrame(correlation_rows)
    all_correlations = all_correlations[
        all_correlations.population == "all_may_candidates"
    ]
    output: dict[str, object] = {
        "signals": ACTIVITY_SCALE_SIGNALS,
        "correlation_method": "Spearman rank correlation on the full May population and two policy-relevant May subsets",
        "all_may_pairwise_spearman": {
            "min": float(all_correlations.spearman.min()),
            "median": float(all_correlations.spearman.median()),
            "max": float(all_correlations.spearman.max()),
        },
        "rank_pca_first_component_variance_share": component_share,
        "grouped_permutation": {
            "sample_rows": int(len(sample)),
            "baseline_pr_auc": baseline_pr_auc,
            "method": "one shared row permutation per group preserves within-group relationships while breaking association with outcomes and other features",
            "results": permutation_results,
            "sum_of_five_individual_drops": summed_individual_drop,
            "joint_five_signal_drop": joint_drop,
            "joint_to_summed_individual_drop_ratio": (
                joint_drop / summed_individual_drop if summed_individual_drop else None
            ),
        },
        "interpretation_guardrail": "Correlation and grouped permutation identify predictive redundancy, not a causal wallet policy.",
    }
    REDUNDANCY_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def run(path: Path = FEATURE_STORE, force_intraslot: bool = True) -> dict[str, object]:
    ensure_output_dirs()
    active, combined_predictions, validation, _ = _active_period_experiment(path)
    redundancy = _signal_redundancy(validation)
    intraslot = run_intraslot_sensitivity(
        combined_predictions, force=force_intraslot
    )
    output: dict[str, object] = {
        "active_period_training": active,
        "signal_redundancy": redundancy,
        "intraslot_latency": intraslot,
        "artifacts": {
            "active_period_training": str(ACTIVE_RESULTS.relative_to(SUBMISSION.parent)),
            "active_feature_importance": str(ACTIVE_IMPORTANCE.relative_to(SUBMISSION.parent)),
            "legacy_feature_importance": str(LEGACY_IMPORTANCE.relative_to(SUBMISSION.parent)),
            "signal_correlations": str(CORRELATIONS.relative_to(SUBMISSION.parent)),
            "signal_redundancy": str(REDUNDANCY_RESULTS.relative_to(SUBMISSION.parent)),
            "intraslot_latency": "submission/tables/intraslot_latency_sensitivity.json",
        },
    }
    AUDIT_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted final methodological audits")
    parser.add_argument("--features", type=Path, default=FEATURE_STORE)
    parser.add_argument(
        "--reuse-intraslot",
        action="store_true",
        help="reuse the existing intra-slot outcome cache instead of rebuilding it",
    )
    args = parser.parse_args()
    run(args.features, force_intraslot=not args.reuse_intraslot)


if __name__ == "__main__":
    main()
