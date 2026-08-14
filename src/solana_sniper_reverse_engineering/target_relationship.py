from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import (
    ACTIVE_ERA_START,
    ARTIFACTS,
    INTERIM,
    JULY_START,
    JUNE_START,
    MAY_START,
    SUBMISSION,
    TARGET_ACTIVITY,
    ensure_output_dirs,
)
from .feature_store import FEATURE_STORE
from .modeling import (
    CATEGORICAL_FEATURES,
    ModelBundle,
    _available_features,
    _thresholds,
    fit_lightgbm,
    metrics,
    predict,
)
from .third_pass import (
    DEV_SELL_FEATURES,
    QUALITY_FEATURES,
    _rank_summary,
    _selected_historical_features,
    audit_dev_sell_features,
    audit_dev_sell_tie_contract,
)


APRIL_START = 1_775_001_600
TRAINING_NEGATIVE_STRIDE = 2

TARGET_SIGNER_EVENTS_PRE_JUNE = INTERIM / "target_signer_buy_events_pre_june.parquet"
TARGET_SIGNER_EVENTS_FULL = INTERIM / "target_signer_buy_events_full.parquet"
TARGET_RELATIONSHIP_PRE_JUNE = INTERIM / "target_signer_relationship_features_pre_june.parquet"
TARGET_RELATIONSHIP_PRE_JUNE_LAG5 = INTERIM / "target_signer_relationship_features_pre_june_lag5.parquet"
TARGET_RELATIONSHIP_FULL = INTERIM / "target_signer_relationship_features_full.parquet"

RESCUE_MODEL_DIR = ARTIFACTS / "models" / "target_relationship_rescue"
VALIDATION_RESULTS = SUBMISSION / "tables" / "target_relationship_validation.json"
JUNE_RESULTS = SUBMISSION / "tables" / "target_relationship_june_reporting.json"
PREDICTIONS = ARTIFACTS / "tables" / "target_relationship_june_predictions.parquet"
FREEZE_MANIFEST = RESCUE_MODEL_DIR / "freeze_manifest.json"


CORE_RELATIONSHIP_FEATURES = [
    "target_signer_known",
    "prior_target_buy_count",
    "prior_target_buy_count_1h",
    "prior_target_buy_count_6h",
    "prior_target_buy_count_1d",
    "prior_target_buy_count_7d",
    "prior_target_buy_count_30d",
    "seconds_since_prior_target_buy",
    "prior_target_buy_fraction",
    "prior_target_buy_rate_shrunk_5",
    "prior_target_buy_rate_shrunk_20",
    "prior_target_buy_count_log1p",
    "prior_target_buy_recency_log1p",
]

EXPANDED_RELATIONSHIP_FEATURES = [
    "deployments_since_prior_target_buy",
    "seconds_since_first_target_buy",
    "prior_target_buy_rate_1h",
    "prior_target_buy_rate_1d",
    "prior_target_buy_rate_7d",
    "prior_target_buy_rate_30d",
    "prior_target_buys_per_known_day",
    "prior_target_deployments_per_buy",
]

ALL_RELATIONSHIP_FEATURES = CORE_RELATIONSHIP_FEATURES + EXPANDED_RELATIONSHIP_FEATURES
DEVELOPER_SELL_TIE_CONTRACT = {
    "id": "latest-prior-launch-group-v1",
    "group_key": ["wallet", "launch_time"],
    "eligibility": "launch_time < candidate block_time",
    "features": {
        "latest_prior_launch_group_dev_sold_fraction": (
            "fraction of tied launches with a developer sell observed strictly before the candidate"
        ),
        "latest_prior_launch_group_mature_1d_no_dev_sell_fraction": (
            "after group age exceeds one day, fraction without a developer sell in the first day"
        ),
        "latest_prior_launch_group_mature_7d_no_dev_sell_fraction": (
            "after group age exceeds seven days, fraction without a developer sell in the first seven days"
        ),
        "latest_prior_launch_group_dev_sell_latency_median_seconds": (
            "median latency among tied launches whose developer sell is observed strictly before the candidate"
        ),
    },
    "ordering_excluded": ["transaction hash", "file order"],
}


def _sql(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _connection(memory: str = "30GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    con.execute("SET threads=20")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    temp = INTERIM / "duckdb_temp_target_relationship"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{_sql(temp)}'")
    return con


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _source_code_sha256() -> dict[str, str]:
    """Hash every local module that defines the corrected Part 2 result."""
    directory = Path(__file__).parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (
            "config.py",
            "feature_store.py",
            "message_features.py",
            "modeling.py",
            "target_relationship.py",
            "third_pass.py",
        )
    }


def _paths_for_cutoff(cutoff: int, confirmation_lag_seconds: int) -> tuple[Path, Path]:
    if cutoff == JUNE_START and confirmation_lag_seconds == 0:
        return TARGET_SIGNER_EVENTS_PRE_JUNE, TARGET_RELATIONSHIP_PRE_JUNE
    if cutoff == JUNE_START and confirmation_lag_seconds == 5:
        return TARGET_SIGNER_EVENTS_PRE_JUNE, TARGET_RELATIONSHIP_PRE_JUNE_LAG5
    if cutoff == JULY_START and confirmation_lag_seconds == 0:
        return TARGET_SIGNER_EVENTS_FULL, TARGET_RELATIONSHIP_FULL
    raise ValueError(
        f"unsupported cutoff/lag combination: cutoff={cutoff}, lag={confirmation_lag_seconds}"
    )


def build_target_signer_events(cutoff: int, *, force: bool = False) -> Path:
    """Map raw first target buys to the signer of the corresponding deployment.

    The mapping uses candidate identities only. Class membership is audited but is not
    written as a model field. A target buy becomes state only at its raw activity time.
    """
    events_path, _ = _paths_for_cutoff(cutoff, 0)
    if events_path.exists() and not force:
        return events_path
    events_path.parent.mkdir(parents=True, exist_ok=True)
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH first_target_buys AS (
            SELECT token_address,min(timestamp) AS target_buy_time
            FROM read_parquet('{_sql(TARGET_ACTIVITY)}')
            WHERE event_type='buy' AND timestamp<{cutoff}
            GROUP BY token_address
          )
          SELECT b.token_address,d.tx_signer,d.block_time AS source_deploy_time,
                 b.target_buy_time
          FROM first_target_buys b
          JOIN read_parquet('{_sql(FEATURE_STORE)}') d USING(token_address)
          WHERE d.label=1 AND b.target_buy_time<{cutoff}
        ) TO '{_sql(events_path)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {events_path} in {time.monotonic() - started:.1f}s", flush=True)
    return events_path


def raw_event_audit(cutoff: int) -> dict[str, int]:
    events_path, _ = _paths_for_cutoff(cutoff, 0)
    con = _connection("12GB")
    row = con.execute(
        f"""
        WITH raw_buys AS (
          SELECT token_address,min(timestamp) AS target_buy_time
          FROM read_parquet('{_sql(TARGET_ACTIVITY)}')
          WHERE event_type='buy' AND timestamp<{cutoff}
          GROUP BY token_address
        ), candidate_matches AS (
          SELECT b.*,d.label,d.block_time
          FROM raw_buys b LEFT JOIN read_parquet('{_sql(FEATURE_STORE)}') d USING(token_address)
        ), emitted AS (
          SELECT * FROM read_parquet('{_sql(events_path)}')
        )
        SELECT
          (SELECT count(*) FROM raw_buys) AS raw_buy_tokens,
          (SELECT count(*) FROM candidate_matches WHERE label=1) AS matched_positive_tokens,
          (SELECT count(*) FROM candidate_matches WHERE label=0) AS matched_negative_tokens,
          (SELECT count(*) FROM candidate_matches WHERE label IS NULL) AS unmatched_tokens,
          (SELECT count(*) FROM emitted) AS emitted_events,
          (SELECT count(*)-count(DISTINCT token_address) FROM emitted) AS duplicate_event_tokens,
          (SELECT count(*) FROM candidate_matches
             WHERE label=1 AND target_buy_time<block_time) AS buys_before_source_deployment,
          (SELECT count(*) FROM candidate_matches
             WHERE label=1 AND target_buy_time=block_time) AS buys_at_source_deployment_second,
          (SELECT count(*) FROM candidate_matches
             WHERE label=1 AND target_buy_time>block_time) AS buys_after_source_deployment_second
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    result = {name: int(value) for name, value in zip(names, row, strict=True)}
    if (
        result["matched_negative_tokens"]
        or result["duplicate_event_tokens"]
        or result["buys_before_source_deployment"]
        or result["emitted_events"] != result["matched_positive_tokens"]
    ):
        raise RuntimeError(f"raw target-event audit failed: {result}")
    return result


def build_relationship_features(
    cutoff: int,
    *,
    confirmation_lag_seconds: int = 0,
    force: bool = False,
) -> Path:
    """Build strict online target-wallet × deployment-signer history.

    For a candidate at time t and confirmation lag L, only target buys with
    target_buy_time < t-L contribute. Equality is excluded at one-second resolution.
    """
    events_path, output = _paths_for_cutoff(cutoff, confirmation_lag_seconds)
    if output.exists() and not force:
        return output
    if not events_path.exists() or force:
        build_target_signer_events(cutoff, force=force)
    con = _connection()
    started = time.monotonic()
    lag = int(confirmation_lag_seconds)
    con.execute(
        f"""
        COPY (
          WITH buy_seconds AS (
            SELECT tx_signer,target_buy_time AS buy_time,count(*)::BIGINT AS buys_at_second
            FROM read_parquet('{_sql(events_path)}')
            GROUP BY 1,2
          ), buy_state_base AS (
            SELECT tx_signer,buy_time,
                   sum(buys_at_second) OVER (
                     PARTITION BY tx_signer ORDER BY buy_time
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS cumulative_target_buys,
                   min(buy_time) OVER (PARTITION BY tx_signer) AS first_target_buy_time
            FROM buy_seconds
          ), deploy_seconds AS (
            SELECT tx_signer,block_time AS deploy_time,count(*)::BIGINT AS deployments_at_second
            FROM read_parquet('{_sql(FEATURE_STORE)}')
            WHERE block_time<{cutoff}
            GROUP BY 1,2
          ), deploy_state AS (
            SELECT tx_signer,deploy_time,
                   sum(deployments_at_second) OVER (
                     PARTITION BY tx_signer ORDER BY deploy_time
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS cumulative_deployments
            FROM deploy_seconds
          ), buy_state AS (
            SELECT b.*,coalesce(d.cumulative_deployments,0) AS cumulative_deployments_before_buy
            FROM buy_state_base b
            ASOF LEFT JOIN deploy_state d
              ON b.tx_signer=d.tx_signer AND b.buy_time>d.deploy_time
          ), joined AS (
            SELECT d.token_address,d.block_time,d.prior_deploy_count,
                   d.prior_deploy_count_1h,d.prior_deploy_count_1d,
                   d.prior_deploy_count_7d,d.prior_deploy_count_30d,
                   s.buy_time AS target_relationship_state_time,
                   s.first_target_buy_time,
                   s.cumulative_deployments_before_buy,
                   coalesce(s.cumulative_target_buys,0) AS prior_target_buy_count,
                   coalesce(s.cumulative_target_buys,0)-coalesce(s1h.cumulative_target_buys,0)
                     AS prior_target_buy_count_1h,
                   coalesce(s.cumulative_target_buys,0)-coalesce(s6h.cumulative_target_buys,0)
                     AS prior_target_buy_count_6h,
                   coalesce(s.cumulative_target_buys,0)-coalesce(s1d.cumulative_target_buys,0)
                     AS prior_target_buy_count_1d,
                   coalesce(s.cumulative_target_buys,0)-coalesce(s7d.cumulative_target_buys,0)
                     AS prior_target_buy_count_7d,
                   coalesce(s.cumulative_target_buys,0)-coalesce(s30d.cumulative_target_buys,0)
                     AS prior_target_buy_count_30d,
                   d.block_time-s.buy_time AS seconds_since_prior_target_buy
            FROM read_parquet('{_sql(FEATURE_STORE)}') d
            ASOF LEFT JOIN buy_state s
              ON d.tx_signer=s.tx_signer AND d.block_time-{lag}>s.buy_time
            ASOF LEFT JOIN buy_state s1h
              ON d.tx_signer=s1h.tx_signer AND d.block_time-{lag}-3600>s1h.buy_time
            ASOF LEFT JOIN buy_state s6h
              ON d.tx_signer=s6h.tx_signer AND d.block_time-{lag}-21600>s6h.buy_time
            ASOF LEFT JOIN buy_state s1d
              ON d.tx_signer=s1d.tx_signer AND d.block_time-{lag}-86400>s1d.buy_time
            ASOF LEFT JOIN buy_state s7d
              ON d.tx_signer=s7d.tx_signer AND d.block_time-{lag}-604800>s7d.buy_time
            ASOF LEFT JOIN buy_state s30d
              ON d.tx_signer=s30d.tx_signer AND d.block_time-{lag}-2592000>s30d.buy_time
            WHERE d.block_time<{cutoff}
          )
          SELECT token_address,target_relationship_state_time,first_target_buy_time,
                 (prior_target_buy_count>0)::UTINYINT AS target_signer_known,
                 prior_target_buy_count,
                 prior_target_buy_count_1h,
                 prior_target_buy_count_6h,
                 prior_target_buy_count_1d,
                 prior_target_buy_count_7d,
                 prior_target_buy_count_30d,
                 seconds_since_prior_target_buy,
                 CASE WHEN prior_deploy_count>0
                      THEN least(prior_target_buy_count/prior_deploy_count,1.0)
                      ELSE 0 END AS prior_target_buy_fraction,
                 prior_target_buy_count/(prior_deploy_count+5.0)
                   AS prior_target_buy_rate_shrunk_5,
                 prior_target_buy_count/(prior_deploy_count+20.0)
                   AS prior_target_buy_rate_shrunk_20,
                 ln(1+prior_target_buy_count) AS prior_target_buy_count_log1p,
                 CASE WHEN seconds_since_prior_target_buy IS NOT NULL
                      THEN ln(1+seconds_since_prior_target_buy) END
                   AS prior_target_buy_recency_log1p,
                 CASE WHEN prior_target_buy_count>0
                      THEN greatest(prior_deploy_count-cumulative_deployments_before_buy,0)
                      ELSE NULL END AS deployments_since_prior_target_buy,
                 CASE WHEN first_target_buy_time IS NOT NULL
                      THEN block_time-first_target_buy_time END AS seconds_since_first_target_buy,
                 least(prior_target_buy_count_1h/greatest(prior_deploy_count_1h,1.0),1.0)
                   AS prior_target_buy_rate_1h,
                 least(prior_target_buy_count_1d/greatest(prior_deploy_count_1d,1.0),1.0)
                   AS prior_target_buy_rate_1d,
                 least(prior_target_buy_count_7d/greatest(prior_deploy_count_7d,1.0),1.0)
                   AS prior_target_buy_rate_7d,
                 least(prior_target_buy_count_30d/greatest(prior_deploy_count_30d,1.0),1.0)
                   AS prior_target_buy_rate_30d,
                 CASE WHEN first_target_buy_time IS NOT NULL
                      THEN prior_target_buy_count/greatest((block_time-first_target_buy_time)/86400.0,1.0)
                      ELSE 0 END AS prior_target_buys_per_known_day,
                 CASE WHEN prior_target_buy_count>0
                      THEN prior_deploy_count/prior_target_buy_count
                      ELSE NULL END AS prior_target_deployments_per_buy
          FROM joined
        ) TO '{_sql(output)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {output} in {time.monotonic() - started:.1f}s", flush=True)
    return output


def audit_relationship_features(path: Path, cutoff: int, confirmation_lag_seconds: int) -> dict[str, int]:
    con = _connection("16GB")
    row = con.execute(
        f"""
        SELECT count(*) AS rows,count(DISTINCT r.token_address) AS tokens,
               count(*)-count(DISTINCT r.token_address) AS duplicate_tokens,
               count(*) FILTER (
                 WHERE target_relationship_state_time IS NOT NULL
                   AND target_relationship_state_time>=f.block_time-{confirmation_lag_seconds}
               ) AS future_or_equal_states,
               count(*) FILTER (WHERE seconds_since_prior_target_buy<=0) AS invalid_recencies,
               count(*) FILTER (
                 WHERE first_target_buy_time IS NOT NULL
                   AND first_target_buy_time>=f.block_time-{confirmation_lag_seconds}
               ) AS invalid_first_buy_times,
               count(*) FILTER (
                 WHERE prior_target_buy_count<0
                    OR prior_target_buy_count_1h<0
                    OR prior_target_buy_count_1h>prior_target_buy_count_6h
                    OR prior_target_buy_count_6h>prior_target_buy_count_1d
                    OR prior_target_buy_count_1d>prior_target_buy_count_7d
                    OR prior_target_buy_count_7d>prior_target_buy_count_30d
                    OR prior_target_buy_count_30d>prior_target_buy_count
                    OR prior_target_buy_count>f.prior_deploy_count
               ) AS invalid_counts,
               count(*) FILTER (
                 WHERE prior_target_buy_fraction NOT BETWEEN 0 AND 1
                    OR prior_target_buy_rate_1h NOT BETWEEN 0 AND 1
                    OR prior_target_buy_rate_1d NOT BETWEEN 0 AND 1
                    OR prior_target_buy_rate_7d NOT BETWEEN 0 AND 1
                    OR prior_target_buy_rate_30d NOT BETWEEN 0 AND 1
               ) AS invalid_rates,
               count(*) FILTER (
                 WHERE deployments_since_prior_target_buy<0
                    OR deployments_since_prior_target_buy>f.prior_deploy_count
               ) AS invalid_deployments_since_buy
        FROM read_parquet('{_sql(path)}') r
        JOIN read_parquet('{_sql(FEATURE_STORE)}') f USING(token_address)
        WHERE f.block_time<{cutoff}
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    result = {name: int(value) for name, value in zip(names, row, strict=True)}
    expected = con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql(FEATURE_STORE)}') WHERE block_time<{cutoff}"
    ).fetchone()[0]
    result["expected_rows"] = int(expected)
    invalid = [
        "duplicate_tokens",
        "future_or_equal_states",
        "invalid_recencies",
        "invalid_first_buy_times",
        "invalid_counts",
        "invalid_rates",
        "invalid_deployments_since_buy",
    ]
    if result["rows"] != result["expected_rows"] or any(result[name] for name in invalid):
        raise RuntimeError(f"target-relationship temporal audit failed: {result}")
    return result


def _load_frame(
    relationship_path: Path,
    predicate: str,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    base_names = set(duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{_sql(FEATURE_STORE)}')").fetchdf().column_name)
    relationship_names = set(duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{_sql(relationship_path)}')").fetchdf().column_name)
    quality_names = set(duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{_sql(QUALITY_FEATURES)}')").fetchdf().column_name)
    sell_names = set(duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{_sql(DEV_SELL_FEATURES)}')").fetchdf().column_name)
    aliases = []
    for name in numeric + categorical:
        if name in base_names:
            aliases.append(f'f."{name}"')
        elif name in relationship_names:
            aliases.append(f'r."{name}"')
        elif name in quality_names:
            aliases.append(f'q."{name}"')
        elif name in sell_names:
            aliases.append(f's."{name}"')
        else:
            raise KeyError(f"feature is unavailable: {name}")
    columns = ["f.token_address", "f.tx_hash", "f.tx_signer", "f.block_time", "f.label", *aliases]
    con = _connection()
    return con.execute(
        f"""
        SELECT {','.join(columns)}
        FROM read_parquet('{_sql(FEATURE_STORE)}') f
        JOIN read_parquet('{_sql(relationship_path)}') r USING(token_address)
        JOIN read_parquet('{_sql(QUALITY_FEATURES)}') q USING(token_address)
        JOIN read_parquet('{_sql(DEV_SELL_FEATURES)}') s USING(token_address)
        WHERE {predicate}
        ORDER BY f.block_time,f.token_address
        """
    ).fetch_df()


def _model_summary(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[dict[str, object], np.ndarray, float]:
    y = frame.label.to_numpy(dtype=np.uint8)
    score = predict(bundle, frame)
    threshold = _thresholds(y, score)["max_f1"]
    result: dict[str, object] = metrics(y, score, threshold)
    result["top_k"] = _rank_summary(y, score)
    return result, score, threshold


def _daily_stability(
    validation: pd.DataFrame,
    baseline_score: np.ndarray,
    rescue_score: np.ndarray,
) -> dict[str, object]:
    dates = pd.to_datetime(validation.block_time, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    rows = []
    for date in sorted(dates.unique()):
        mask = dates.eq(date).to_numpy()
        y = validation.loc[mask, "label"].to_numpy(dtype=np.uint8)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        before = float(average_precision_score(y, baseline_score[mask]))
        after = float(average_precision_score(y, rescue_score[mask]))
        rows.append({"date": date, "baseline_pr_auc": before, "rescue_pr_auc": after, "delta": after - before})
    delta = np.asarray([item["delta"] for item in rows], dtype=float)
    return {
        "days": len(rows),
        "positive_delta_days": int((delta > 0).sum()),
        "positive_delta_share": float((delta > 0).mean()),
        "mean_daily_delta": float(delta.mean()),
        "median_daily_delta": float(np.median(delta)),
        "rows": rows,
    }


def _window_experiment(
    period: str,
    relationship_path: Path,
    train_end: int,
    validation_start: int,
    validation_end: int,
) -> tuple[dict[str, object], dict[str, ModelBundle], dict[str, float]]:
    base_numeric, categorical = _available_features(FEATURE_STORE)
    selected_outcome_features, _ = _selected_historical_features()
    final_numeric = base_numeric + selected_outcome_features
    maximum_numeric = final_numeric + ALL_RELATIONSHIP_FEATURES
    train_predicate = (
        f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{train_end} "
        f"AND (f.label=1 OR hash(f.token_address)%{TRAINING_NEGATIVE_STRIDE}=0)"
    )
    validation_predicate = f"f.block_time>={validation_start} AND f.block_time<{validation_end}"
    train = _load_frame(relationship_path, train_predicate, maximum_numeric, categorical)
    validation = _load_frame(relationship_path, validation_predicate, maximum_numeric, categorical)
    feature_sets = {
        "preserved_control": base_numeric,
        "preserved_final": final_numeric,
        "relationship_only": CORE_RELATIONSHIP_FEATURES,
        "final_plus_core_relationship": final_numeric + CORE_RELATIONSHIP_FEATURES,
        "final_plus_expanded_relationship": maximum_numeric,
    }
    models: dict[str, ModelBundle] = {}
    summaries: dict[str, object] = {
        "train_rows": int(len(train)),
        "train_positives": int(train.label.sum()),
        "validation_rows": int(len(validation)),
        "validation_positives": int(validation.label.sum()),
    }
    scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for name, numeric in feature_sets.items():
        model = fit_lightgbm(train, numeric, categorical if name != "relationship_only" else [], TRAINING_NEGATIVE_STRIDE)
        summary, score, threshold = _model_summary(model, validation)
        models[name] = model
        summaries[name] = summary
        scores[name] = score
        thresholds[name] = threshold
    summaries["core_minus_preserved_final_pr_auc"] = float(
        summaries["final_plus_core_relationship"]["pr_auc"] - summaries["preserved_final"]["pr_auc"]
    )
    summaries["expanded_minus_core_pr_auc"] = float(
        summaries["final_plus_expanded_relationship"]["pr_auc"]
        - summaries["final_plus_core_relationship"]["pr_auc"]
    )
    summaries["daily_stability_vs_preserved_final"] = _daily_stability(
        validation,
        scores["preserved_final"],
        scores["final_plus_core_relationship"],
    )
    print(period, json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    return summaries, models, thresholds


def _single_variant_window(
    relationship_path: Path,
    train_end: int,
    validation_start: int,
    validation_end: int,
    relationship_features: list[str],
) -> dict[str, object]:
    """Fit the frozen candidate feature set on a temporal-sensitivity state table."""
    base_numeric, categorical = _available_features(FEATURE_STORE)
    selected_outcome_features, _ = _selected_historical_features()
    numeric = base_numeric + selected_outcome_features + relationship_features
    train = _load_frame(
        relationship_path,
        f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{train_end} "
        f"AND (f.label=1 OR hash(f.token_address)%{TRAINING_NEGATIVE_STRIDE}=0)",
        numeric,
        categorical,
    )
    validation = _load_frame(
        relationship_path,
        f"f.block_time>={validation_start} AND f.block_time<{validation_end}",
        numeric,
        categorical,
    )
    model = fit_lightgbm(train, numeric, categorical, TRAINING_NEGATIVE_STRIDE)
    summary, _, _ = _model_summary(model, validation)
    return summary


def run_validation(*, force_features: bool = False) -> dict[str, object]:
    """Run pre-June gates only and freeze an accepted design without scoring June."""
    ensure_output_dirs()
    RESCUE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    relationship_path = build_relationship_features(JUNE_START, force=force_features)
    event_audit = raw_event_audit(JUNE_START)
    temporal_audit = audit_relationship_features(relationship_path, JUNE_START, 0)
    developer_sell_tie_audit = audit_dev_sell_tie_contract()
    developer_sell_feature_audit = audit_dev_sell_features()
    windows: dict[str, object] = {}
    fitted: dict[str, dict[str, ModelBundle]] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for period, train_end, start, end in (
        ("april", APRIL_START, APRIL_START, MAY_START),
        ("may", MAY_START, MAY_START, JUNE_START),
    ):
        windows[period], fitted[period], thresholds[period] = _window_experiment(
            period, relationship_path, train_end, start, end
        )

    core_passes = all(
        float(windows[period]["final_plus_core_relationship"]["pr_auc"])
        >= (0.20 if period == "april" else 0.30)
        and float(windows[period]["core_minus_preserved_final_pr_auc"]) >= 0.05
        for period in ("april", "may")
    )
    expanded_passes = all(
        float(windows[period]["expanded_minus_core_pr_auc"]) >= 0.005
        for period in ("april", "may")
    )
    selected_name = (
        "final_plus_expanded_relationship" if expanded_passes else "final_plus_core_relationship"
    )
    selected_relationship_features = (
        ALL_RELATIONSHIP_FEATURES if expanded_passes else CORE_RELATIONSHIP_FEATURES
    )
    lag_path = build_relationship_features(
        JUNE_START,
        confirmation_lag_seconds=5,
        force=force_features,
    )
    lag_audit = audit_relationship_features(lag_path, JUNE_START, 5)
    lag_windows: dict[str, object] = {}
    for period, train_end, start, end in (
        ("april", APRIL_START, APRIL_START, MAY_START),
        ("may", MAY_START, MAY_START, JUNE_START),
    ):
        lag_summary = _single_variant_window(
            lag_path,
            train_end,
            start,
            end,
            selected_relationship_features,
        )
        zero_lag_ap = float(windows[period][selected_name]["pr_auc"])
        lag_summary["pr_auc_delta_vs_zero_lag"] = float(lag_summary["pr_auc"] - zero_lag_ap)
        lag_windows[period] = lag_summary
    lag_resilient = all(
        float(lag_windows[period]["pr_auc_delta_vs_zero_lag"]) >= -0.03
        for period in ("april", "may")
    )
    decision = "PROMOTE" if core_passes and lag_resilient else "REJECT"
    output: dict[str, object] = {
        "status": decision,
        "decision_clock": "A raw target buy contributes only when target_buy_time < candidate block_time; equality is excluded.",
        "source": str(TARGET_ACTIVITY.relative_to(TARGET_ACTIVITY.parents[3])),
        "feature_path": str(relationship_path),
        "event_audit": event_audit,
        "temporal_audit": temporal_audit,
        "developer_sell_tie_contract": DEVELOPER_SELL_TIE_CONTRACT,
        "developer_sell_tie_audit": developer_sell_tie_audit,
        "developer_sell_feature_audit": developer_sell_feature_audit,
        "feature_sets": {
            "core": CORE_RELATIONSHIP_FEATURES,
            "expanded_candidates": EXPANDED_RELATIONSHIP_FEATURES,
            "selected_model": selected_name if core_passes else None,
        },
        "selection_rule": {
            "core": "PROMOTE only for AP >=0.20 April, >=0.30 May, and >=0.05 AP lift over the preserved final on both windows.",
            "expanded": "Use expanded features only for >=0.005 AP lift over core on both windows.",
            "confirmation_lag": "Require no worse than a 0.03 absolute AP loss in either window after excluding target buys from the prior five seconds.",
        },
        "confirmation_lag_5_seconds": {
            "temporal_audit": lag_audit,
            "windows": lag_windows,
        },
        "windows": windows,
        "june_opened": False,
    }
    VALIDATION_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if decision == "PROMOTE":
        selected_model = fitted["may"][selected_name]
        selected_threshold = thresholds["may"][selected_name]
        joblib.dump(selected_model.model, RESCUE_MODEL_DIR / "model.joblib")
        model_features = {
            "numeric_features": selected_model.numeric_features,
            "categorical_features": selected_model.categorical_features,
            "transformed_feature_names": selected_model.feature_names_out,
        }
        (RESCUE_MODEL_DIR / "model_features.json").write_text(
            json.dumps(model_features, indent=2, sort_keys=True) + "\n"
        )
        freeze = {
            "status": "FROZEN_PRE_JUNE",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "module_sha256": _module_sha256(),
            "source_code_sha256": _source_code_sha256(),
            "developer_sell_tie_contract": DEVELOPER_SELL_TIE_CONTRACT,
            "validation_results_sha256": hashlib.sha256(VALIDATION_RESULTS.read_bytes()).hexdigest(),
            "selected_model": selected_name,
            "selected_relationship_features": selected_relationship_features,
            "numeric_features": selected_model.numeric_features,
            "categorical_features": selected_model.categorical_features,
            "threshold_selected_on_may": selected_threshold,
            "training_window": "2026-03-12 through 2026-04-30",
            "validation_window": "2026-05-01 through 2026-05-31",
            "june_scored": False,
        }
        FREEZE_MANIFEST.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def _june_subgroups(frame: pd.DataFrame, score: np.ndarray, threshold: float) -> dict[str, object]:
    output: dict[str, object] = {}
    for name, mask in (
        ("known_signer", frame.target_signer_known.eq(1).to_numpy()),
        ("cold_signer", frame.target_signer_known.eq(0).to_numpy()),
    ):
        y = frame.loc[mask, "label"].to_numpy(dtype=np.uint8)
        output[name] = metrics(y, score[mask], threshold)
    dates = pd.to_datetime(frame.block_time, unit="s", utc=True)
    week = dates.dt.to_period("W-SUN").astype(str)
    weekly = []
    for value in sorted(week.unique()):
        mask = week.eq(value).to_numpy()
        y = frame.loc[mask, "label"].to_numpy(dtype=np.uint8)
        weekly.append({"week": value, **metrics(y, score[mask], threshold)})
    output["weekly"] = weekly
    return output


def run_june_reporting(*, force_features: bool = False) -> dict[str, object]:
    """Score June once using the immutable pre-June freeze manifest and model."""
    if not FREEZE_MANIFEST.exists() or not VALIDATION_RESULTS.exists():
        raise RuntimeError("run the pre-June validation stage before June reporting")
    if JUNE_RESULTS.exists():
        raise RuntimeError(f"June has already been scored: {JUNE_RESULTS}")
    freeze = json.loads(FREEZE_MANIFEST.read_text())
    if freeze.get("status") != "FROZEN_PRE_JUNE" or freeze.get("june_scored") is not False:
        raise RuntimeError(f"invalid freeze manifest: {freeze}")
    if freeze["module_sha256"] != _module_sha256():
        raise RuntimeError("target_relationship.py changed after the pre-June design freeze")
    if freeze.get("source_code_sha256") != _source_code_sha256():
        raise RuntimeError("Part 2 source code changed after the pre-June design freeze")
    validation_hash = hashlib.sha256(VALIDATION_RESULTS.read_bytes()).hexdigest()
    if freeze["validation_results_sha256"] != validation_hash:
        raise RuntimeError("pre-June validation results changed after the design freeze")

    relationship_path = build_relationship_features(JULY_START, force=force_features)
    event_audit = raw_event_audit(JULY_START)
    temporal_audit = audit_relationship_features(relationship_path, JULY_START, 0)
    numeric = list(freeze["numeric_features"])
    categorical = list(freeze["categorical_features"])
    june = _load_frame(
        relationship_path,
        f"f.block_time>={JUNE_START} AND f.block_time<{JULY_START}",
        numeric,
        categorical,
    )
    pipeline = joblib.load(RESCUE_MODEL_DIR / "model.joblib")
    bundle = ModelBundle(
        pipeline,
        numeric,
        categorical,
        list(freeze.get("transformed_feature_names", [])),
        str(freeze["selected_model"]),
    )
    score = predict(bundle, june)
    threshold = float(freeze["threshold_selected_on_may"])
    y = june.label.to_numpy(dtype=np.uint8)
    output = {
        "status": "FROZEN_JUNE_REPORT",
        "design_source": str(FREEZE_MANIFEST),
        "selected_model": freeze["selected_model"],
        "threshold": threshold,
        "event_audit": event_audit,
        "temporal_audit": temporal_audit,
        "metrics": {**metrics(y, score, threshold), "top_k": _rank_summary(y, score)},
        "subgroups": _june_subgroups(june, score, threshold),
        "no_post_june_redesign": True,
    }
    pd.DataFrame(
        {
            "token_address": june.token_address,
            "block_time": june.block_time,
            "label": june.label.astype("uint8"),
            "score": score,
            "selected": (score >= threshold).astype("uint8"),
        }
    ).to_parquet(PREDICTIONS, index=False)
    JUNE_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict online target-wallet × deployment-signer rescue")
    parser.add_argument("stage", choices=("validate", "june"))
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    if args.stage == "validate":
        run_validation(force_features=args.force_features)
    else:
        run_june_reporting(force_features=args.force_features)


if __name__ == "__main__":
    main()
