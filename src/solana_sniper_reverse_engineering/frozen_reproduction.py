from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from .backtest import (
    _actual_target_june,
    _build_outcomes,
    _metrics as backtest_metrics,
    _strategy_parameters,
)
from .config import (
    ACTIVE_ERA_START,
    ARTIFACTS,
    BOUGHT_ACTIVITY,
    BOUGHT_INDEX,
    BOUGHT_TXS,
    INTERIM,
    JULY_START,
    JUNE_START,
    JUNE_TRADES,
    MAY_START,
    NOT_BOUGHT_ACTIVITY,
    NOT_BOUGHT_INDEX,
    NOT_BOUGHT_TXS,
    PROCESSED,
    ROOT,
    SUBMISSION,
    TARGET_ACTIVITY,
    TARGET_TXS,
    TARGET_TX_INDEX,
    ensure_output_dirs,
)
from .feature_store import FEATURE_STORE, audit as audit_feature_store
from .message_features import run as extract_message_features
from .modeling import (
    ModelBundle,
    _available_features,
    _load_split,
    _preprocessor,
    fit_lightgbm,
    metrics,
    predict,
)
from .target_relationship import (
    TARGET_RELATIONSHIP_FULL,
    audit_relationship_features,
    build_relationship_features,
)
from .third_pass import (
    DEV_SELL_FEATURES,
    QUALITY_FEATURES,
    _economic_frame,
    _joined_frame,
    _two_stage_selection,
    audit_dev_sell_features,
    audit_dev_sell_tie_contract,
    audit_quality_features,
    build_claim_outcomes,
    build_dev_sell_features,
)


RECIPE = SUBMISSION / "tables" / "target_relationship_reproduction_recipe.json"
REPRODUCTION_FEATURES = PROCESSED / "frozen_part2_reproduction_features.parquet"
REPRODUCTION_FEATURE_MANIFEST = REPRODUCTION_FEATURES.with_suffix(".manifest.json")
REPRODUCTION_DIR = ARTIFACTS / "reproduction"
REPRODUCTION_MODEL = REPRODUCTION_DIR / "target_relationship_model.joblib"
REPRODUCTION_PREDICTIONS = REPRODUCTION_DIR / "target_relationship_june_predictions.parquet"
REPRODUCTION_BACKTEST_OUTCOMES = REPRODUCTION_DIR / "target_relationship_backtest_outcomes.parquet"
REPRODUCTION_REPORT = REPRODUCTION_DIR / "target_relationship_reproduction_report.json"
PRIMARY_PART3_RESULTS = SUBMISSION / "tables" / "target_relationship_primary_backtest.json"
EXACT_REPLAY_PREDICTIONS = REPRODUCTION_DIR / "frozen_exact_strategy_predictions.parquet"
EXACT_REPLAY_OUTCOMES = REPRODUCTION_DIR / "frozen_exact_curve_outcomes.parquet"
EXACT_REPLAY_REPORT = REPRODUCTION_DIR / "frozen_exact_replay_publication_audit.json"
TRAINING_NEGATIVE_STRIDE = 2


def _sql(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recipe() -> dict[str, object]:
    if not RECIPE.exists():
        raise FileNotFoundError(
            f"Missing public frozen recipe {RECIPE}. Run scripts/build_submission.py locally."
        )
    recipe = json.loads(RECIPE.read_text())
    if recipe.get("recipe_status") != "FROZEN_REPRODUCTION_ONLY":
        raise RuntimeError(f"invalid frozen reproduction recipe: {recipe.get('recipe_status')}")
    return recipe


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "joblib": joblib.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _feature_sources(
    numeric: list[str], categorical: list[str]
) -> tuple[list[str], list[str]]:
    descriptions = {
        "f": set(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql(FEATURE_STORE)}')"
            ).fetchdf().column_name
        ),
        "q": set(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql(QUALITY_FEATURES)}')"
            ).fetchdf().column_name
        ),
        "s": set(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql(DEV_SELL_FEATURES)}')"
            ).fetchdf().column_name
        ),
        "r": set(
            duckdb.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql(TARGET_RELATIONSHIP_FULL)}')"
            ).fetchdf().column_name
        ),
    }
    aliases: list[str] = []
    for name in numeric + categorical:
        matches = [alias for alias, columns in descriptions.items() if name in columns]
        if len(matches) != 1:
            raise RuntimeError(f"feature {name!r} has source matches {matches}")
        aliases.append(f'{matches[0]}."{name}"')
    return aliases, ["f.token_address", "f.tx_hash", "f.tx_signer", "f.block_time", "f.label"]


def _raw_source_inventory() -> list[dict[str, object]]:
    paths = (
        BOUGHT_TXS,
        BOUGHT_INDEX,
        BOUGHT_ACTIVITY,
        NOT_BOUGHT_TXS,
        NOT_BOUGHT_INDEX,
        NOT_BOUGHT_ACTIVITY,
        TARGET_ACTIVITY,
        TARGET_TXS,
        TARGET_TX_INDEX,
        JUNE_TRADES,
    )
    return [
        {
            "source": str(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _cache_counts(path: Path) -> dict[str, int]:
    con = duckdb.connect()
    row = con.execute(
        f"""
        SELECT count(*) AS rows,count(DISTINCT token_address) AS tokens,
               count(*)-count(DISTINCT token_address) AS duplicate_tokens,
               count(*) FILTER (
                 WHERE block_time>={ACTIVE_ERA_START} AND block_time<{MAY_START}
               ) AS training_population_rows,
               count(*) FILTER (
                 WHERE block_time>={ACTIVE_ERA_START} AND block_time<{MAY_START}
                   AND label=1
               ) AS training_population_positives,
               count(*) FILTER (
                 WHERE block_time>={ACTIVE_ERA_START} AND block_time<{MAY_START}
                   AND (label=1 OR hash(token_address)%{TRAINING_NEGATIVE_STRIDE}=0)
               ) AS sampled_training_rows,
               count(*) FILTER (
                 WHERE block_time>={JUNE_START} AND block_time<{JULY_START}
               ) AS june_rows,
               count(*) FILTER (
                 WHERE block_time>={JUNE_START} AND block_time<{JULY_START} AND label=1
               ) AS june_positives
        FROM read_parquet('{_sql(path)}')
        """
    ).fetchone()
    return {
        name: int(value)
        for name, value in zip((item[0] for item in con.description), row, strict=True)
    }


def _validate_cache(path: Path, manifest_path: Path) -> dict[str, object]:
    recipe = _recipe()
    manifest = json.loads(manifest_path.read_text())
    if manifest["recipe_sha256"] != _sha256(RECIPE):
        raise RuntimeError("reproduction feature cache was built for another frozen recipe")
    if manifest["cache_sha256"] != _sha256(path):
        raise RuntimeError("reproduction feature cache bytes do not match its manifest")
    counts = _cache_counts(path)
    expected = recipe["expected"]["feature_state"]
    for key, value in expected.items():
        if counts[key] != int(value):
            raise RuntimeError(f"reproduction feature cache {key}: {counts[key]} != {value}")
    return {"mode": "verified_private_working_cache", "path": str(path), **counts}


def build_reproduction_feature_cache(
    *,
    force_from_raw: bool = False,
    force_cache: bool = False,
) -> tuple[Path, dict[str, object]]:
    """Construct the exact frozen train/June matrix or verify the public cache.

    The cache is a private working derivative, never a public Resource input. Deleting
    it and setting ``force_from_raw=True`` executes the same source builders from the
    authorized competition files.
    """
    ensure_output_dirs()
    recipe = _recipe()
    if (
        REPRODUCTION_FEATURES.exists()
        and REPRODUCTION_FEATURE_MANIFEST.exists()
        and not force_cache
        and not force_from_raw
    ):
        return REPRODUCTION_FEATURES, _validate_cache(
            REPRODUCTION_FEATURES, REPRODUCTION_FEATURE_MANIFEST
        )

    extract_message_features(force=force_from_raw)
    from .feature_store import run as build_feature_store

    feature_path = build_feature_store(force=force_from_raw)
    base_audit = audit_feature_store(feature_path)
    build_dev_sell_features(force=force_from_raw)
    quality_audit = audit_quality_features()
    sell_audit = audit_dev_sell_features()
    relationship_path = build_relationship_features(
        JULY_START, force=force_from_raw
    )
    relationship_audit = audit_relationship_features(
        relationship_path, JULY_START, 0
    )

    numeric = list(recipe["features"]["numeric"])
    categorical = list(recipe["features"]["categorical"])
    aliases, identifiers = _feature_sources(numeric, categorical)
    columns = identifiers + aliases
    con = duckdb.connect()
    con.execute("SET memory_limit='30GB'")
    con.execute("SET threads=20")
    con.execute("SET preserve_insertion_order=false")
    temp = INTERIM / "duckdb_temp_frozen_reproduction"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{_sql(temp)}'")
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          SELECT {','.join(columns)}
          FROM read_parquet('{_sql(FEATURE_STORE)}') f
          JOIN read_parquet('{_sql(QUALITY_FEATURES)}') q USING(token_address)
          JOIN read_parquet('{_sql(DEV_SELL_FEATURES)}') s USING(token_address)
          JOIN read_parquet('{_sql(TARGET_RELATIONSHIP_FULL)}') r USING(token_address)
          WHERE (f.block_time>={ACTIVE_ERA_START} AND f.block_time<{MAY_START})
             OR (f.block_time>={JUNE_START} AND f.block_time<{JULY_START})
          ORDER BY f.block_time,f.token_address
        ) TO '{_sql(REPRODUCTION_FEATURES)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    counts = _cache_counts(REPRODUCTION_FEATURES)
    expected = recipe["expected"]["feature_state"]
    for key, value in expected.items():
        if counts[key] != int(value):
            raise RuntimeError(f"built reproduction feature cache {key}: {counts[key]} != {value}")
    manifest = {
        "artifact_role": "derived feature cache only; contains no trained model or prediction score",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(REPRODUCTION_FEATURES),
        "cache_bytes": REPRODUCTION_FEATURES.stat().st_size,
        "cache_sha256": _sha256(REPRODUCTION_FEATURES),
        "decision_clock": "strictly prior state; equality excluded",
        "elapsed_seconds": time.monotonic() - started,
        "recipe_sha256": _sha256(RECIPE),
        "raw_sources": _raw_source_inventory(),
        "counts": counts,
        "source_audits": {
            "base": base_audit,
            "quality": quality_audit,
            "developer_sell": sell_audit,
            "developer_sell_tie_contract": audit_dev_sell_tie_contract(),
            "target_relationship": relationship_audit,
        },
    }
    REPRODUCTION_FEATURE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return REPRODUCTION_FEATURES, {"mode": "rebuilt_from_authorized_inputs", **counts}


def _load_reproduction_frame(path: Path, split: str) -> pd.DataFrame:
    recipe = _recipe()
    numeric = list(recipe["features"]["numeric"])
    categorical = list(recipe["features"]["categorical"])
    columns = [
        "token_address",
        "tx_hash",
        "tx_signer",
        "block_time",
        "label",
        *numeric,
        *categorical,
    ]
    if split == "train":
        predicate = (
            f"block_time>={ACTIVE_ERA_START} AND block_time<{MAY_START} "
            f"AND (label=1 OR hash(token_address)%{TRAINING_NEGATIVE_STRIDE}=0)"
        )
    elif split == "june":
        predicate = f"block_time>={JUNE_START} AND block_time<{JULY_START}"
    else:
        raise ValueError(split)
    con = duckdb.connect()
    con.execute("SET memory_limit='30GB'")
    con.execute("SET threads=20")
    selected = ",".join(f'"{column}"' for column in columns)
    return con.execute(
        f"SELECT {selected} FROM read_parquet('{_sql(path)}') "
        f"WHERE {predicate} ORDER BY block_time,token_address"
    ).fetch_df()


def _fit_frozen_model(train: pd.DataFrame, recipe: dict[str, object]) -> ModelBundle:
    numeric = list(recipe["features"]["numeric"])
    categorical = list(recipe["features"]["categorical"])
    expected_preprocessing = {
        "numeric_imputer": {"strategy": "median", "add_indicator": True},
        "categorical_imputer": {"strategy": "most_frequent"},
        "one_hot_encoder": {
            "handle_unknown": "ignore",
            "sparse_output": False,
            "dtype": "float32",
        },
    }
    if recipe["preprocessing"] != expected_preprocessing:
        raise RuntimeError("public recipe preprocessing differs from the frozen implementation")
    estimator = LGBMClassifier(**dict(recipe["lightgbm_parameters"]))
    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale=False)),
            ("model", estimator),
        ]
    )
    labels = train.label.to_numpy(dtype=np.uint8)
    weights = np.where(labels == 1, 1.0, float(TRAINING_NEGATIVE_STRIDE))
    pipeline.fit(
        train[numeric + categorical],
        labels,
        model__sample_weight=weights,
    )
    return ModelBundle(
        pipeline,
        numeric,
        categorical,
        list(pipeline.named_steps["preprocess"].get_feature_names_out()),
        "frozen_target_relationship_reproduction",
    )


def _score_fingerprints(
    token_address: pd.Series,
    labels: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    quantized = np.rint(score * 100_000_000_000).astype("<i8", copy=False)
    selected = score >= threshold
    selected_digest = hashlib.sha256()
    label_selection_digest = hashlib.sha256()
    rank_order = np.lexsort((token_address.astype(str).to_numpy(), -score))
    ranking_digest = hashlib.sha256()
    for index in rank_order:
        ranking_digest.update(str(token_address.iloc[index]).encode())
        ranking_digest.update(b"\n")
    for token, label, is_selected in zip(
        token_address.astype(str), labels, selected, strict=True
    ):
        if is_selected:
            selected_digest.update(token.encode())
            selected_digest.update(b"\n")
        label_selection_digest.update(token.encode())
        label_selection_digest.update(bytes((int(label), int(is_selected))))
    return {
        "score_quantization": "numpy.rint(score * 1e11), little-endian int64, row order block_time/token_address",
        "score_quantized_11_sha256": hashlib.sha256(quantized.tobytes()).hexdigest(),
        "full_ranking_sha256": ranking_digest.hexdigest(),
        "selected_token_sha256": selected_digest.hexdigest(),
        "token_label_selection_sha256": label_selection_digest.hexdigest(),
        "selected_entries": int(selected.sum()),
    }


def strategy_selection_fingerprints(
    predictions: pd.DataFrame,
    selection_columns: tuple[str, ...] = (
        "baseline_selected",
        "quality_selected",
        "two_stage_selected",
        "selective_two_stage_selected",
    ),
) -> dict[str, dict[str, object]]:
    """Fingerprint strategy membership without publishing any candidate rows."""
    ordered = predictions.sort_values("token_address", kind="mergesort")
    labels = ordered.label.to_numpy(dtype=np.uint8)
    output: dict[str, dict[str, object]] = {}
    for column in selection_columns:
        selected = ordered[column].to_numpy(dtype=np.uint8)
        selected_digest = hashlib.sha256()
        label_membership_digest = hashlib.sha256()
        for token, label, is_selected in zip(
            ordered.token_address.astype(str), labels, selected, strict=True
        ):
            if is_selected:
                selected_digest.update(token.encode())
                selected_digest.update(b"\n")
            label_membership_digest.update(token.encode())
            label_membership_digest.update(bytes((int(label), int(is_selected))))
        output[column] = {
            "selected_entries": int(selected.sum()),
            "target_overlap": int((selected * labels).sum()),
            "selected_token_sha256": selected_digest.hexdigest(),
            "token_label_selection_sha256": label_membership_digest.hexdigest(),
        }
    return output


def reproduce_frozen_exact_replay() -> dict[str, object]:
    """Rebuild the frozen economic selections and exact replay from gated inputs.

    Row-level predictions and replay outcomes are written only to the notebook's
    private working directory. The public package carries only this code, the frozen
    recipe, and aggregate reference results.
    """
    ensure_output_dirs()
    REPRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    recipe = _recipe()
    exact = dict(recipe["exact_replay_reproduction"])
    if exact.get("status") != "FROZEN_REPLAY_REPRODUCTION_ONLY":
        raise RuntimeError("invalid frozen exact-replay recipe")
    started = time.monotonic()

    control = dict(exact["control_model"])
    quality = dict(exact["quality_model"])
    economic = dict(exact["economic_model"])
    control_numeric = list(control["numeric_features"])
    control_categorical = list(control["categorical_features"])
    available_numeric, available_categorical = _available_features(FEATURE_STORE)
    if control_numeric != available_numeric or control_categorical != available_categorical:
        raise RuntimeError("frozen control feature list differs from the rebuilt store")

    control_train = _load_split(
        FEATURE_STORE,
        control_numeric,
        control_categorical,
        "train",
        int(control["negative_stride"]),
    )
    if len(control_train) != int(control["expected_training_rows"]):
        raise RuntimeError("frozen control training population changed")
    control_bundle = fit_lightgbm(
        control_train,
        control_numeric,
        control_categorical,
        int(control["negative_stride"]),
    )
    del control_train

    quality_numeric = list(quality["numeric_features"])
    quality_categorical = list(quality["categorical_features"])
    predicate = (
        f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{MAY_START} "
        f"AND (f.label=1 OR hash(f.token_address)%{int(quality['negative_stride'])}=0)"
    )
    quality_train = _joined_frame(
        predicate,
        quality_numeric,
        quality_categorical,
        quality=True,
        dev_sell=True,
    )
    if len(quality_train) != int(quality["expected_training_rows"]):
        raise RuntimeError("frozen quality-model training population changed")
    quality_bundle = fit_lightgbm(
        quality_train,
        quality_numeric,
        quality_categorical,
        int(quality["negative_stride"]),
    )
    del quality_train

    build_claim_outcomes()
    economic_train = _economic_frame(
        int(economic["training_start_unix"]),
        int(economic["training_end_exclusive_unix"]),
        quality_numeric,
        quality_categorical,
        training_stride=int(economic["negative_stride"]),
        dev_sell=True,
    )
    if len(economic_train) != int(economic["expected_training_rows"]):
        raise RuntimeError("frozen economic-model training population changed")
    if int(economic_train.label.sum()) != int(economic["expected_training_positives"]):
        raise RuntimeError("frozen economic-model positive count changed")
    economic_bundle = fit_lightgbm(
        economic_train,
        quality_numeric,
        quality_categorical,
        int(economic["negative_stride"]),
    )
    del economic_train

    june = _joined_frame(
        f"f.block_time>={JUNE_START} AND f.block_time<{JULY_START}",
        quality_numeric,
        quality_categorical,
        quality=True,
        dev_sell=True,
    )
    if len(june) != int(exact["expected_june_rows"]):
        raise RuntimeError("frozen exact-replay June population changed")
    control_score = predict(control_bundle, june)
    quality_score = predict(quality_bundle, june)
    economic_score = predict(economic_bundle, june)
    baseline_selected = control_score >= float(control["threshold"])
    quality_selected = quality_score >= float(quality["threshold"])
    two_stage_selected, combined_score = _two_stage_selection(
        quality_score,
        economic_score,
        int(quality_selected.sum()),
    )
    selective_count = max(
        1,
        int(round(float(exact["selective_trade_count_multiplier"]) * quality_selected.sum())),
    )
    finite = np.flatnonzero(np.isfinite(combined_score))
    selected_indices = finite[
        np.argpartition(combined_score[finite], -selective_count)[-selective_count:]
    ]
    selective_selected = np.zeros(len(june), dtype=bool)
    selective_selected[selected_indices] = True
    predictions = pd.DataFrame(
        {
            "token_address": june.token_address,
            "block_time": june.block_time,
            "label": june.label.astype("uint8"),
            "bot_score": control_score,
            "quality_bot_score": quality_score,
            "economic_score": economic_score,
            "combined_score": combined_score,
            "baseline_selected": baseline_selected.astype("uint8"),
            "quality_selected": quality_selected.astype("uint8"),
            "two_stage_selected": two_stage_selected.astype("uint8"),
            "selective_two_stage_selected": selective_selected.astype("uint8"),
        }
    )
    observed_fingerprints = strategy_selection_fingerprints(predictions)
    predictions.to_parquet(EXACT_REPLAY_PREDICTIONS, index=False)
    if observed_fingerprints != exact["expected_selection_fingerprints"]:
        raise RuntimeError(
            "fresh economic-strategy membership differs from the frozen recipe: "
            + json.dumps(observed_fingerprints, sort_keys=True)
        )
    from .curve_replay import run_curve_replay

    tracked_curve = json.loads(
        (SUBMISSION / "tables" / "curve_replay_results.json").read_text()
    )
    curve = run_curve_replay(
        force=True,
        predictions_path=EXACT_REPLAY_PREDICTIONS,
        outcomes_path=EXACT_REPLAY_OUTCOMES,
        publish_outputs=False,
    )
    reference_match = curve == tracked_curve
    status = (
        "PASS_FRESH_STRATEGY_SELECTION_AND_EXACT_CURVE_REPLAY"
        if reference_match
        else "BLOCKED_STALE_EXACT_REPLAY_REFERENCE"
    )
    selective_fresh = curve["results"]["offset_118"]["selective_two_stage"]
    selective_reference = tracked_curve["results"]["offset_118"][
        "selective_two_stage"
    ]
    result = {
        "status": status,
        "aggregate_reference_match": reference_match,
        "blocker": (
            None
            if reference_match
            else "Fresh replay from the frozen current selection recipe differs from the tracked aggregate. The tracked values depended on a stale row-level outcome cache and cannot be reproduced from the current frozen recipe alone."
        ),
        "selection_fingerprints": observed_fingerprints,
        "fresh_selective_offset_118": selective_fresh,
        "tracked_selective_offset_118": selective_reference,
        "row_level_outputs": {
            "publication_policy": "private notebook working files; never public Resource inputs",
            "predictions": str(EXACT_REPLAY_PREDICTIONS),
            "outcomes": str(EXACT_REPLAY_OUTCOMES),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    EXACT_REPLAY_REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _assert_metrics(
    observed: dict[str, float | int], expected: dict[str, float | int], tolerance: float
) -> None:
    for key, value in expected.items():
        if isinstance(value, int):
            if int(observed[key]) != value:
                raise RuntimeError(f"metric {key}: {observed[key]} != {value}")
        elif not np.isclose(float(observed[key]), float(value), rtol=0.0, atol=tolerance):
            raise RuntimeError(f"metric {key}: {observed[key]} != {value} within {tolerance}")


def reproduce_part3_overlap_and_backtest(
    predictions: pd.DataFrame,
    *,
    force: bool = False,
    aggregate_output: Path | None = None,
) -> dict[str, object]:
    """Recompute the primary Part 3 claim from fresh corrected Part 2 scores."""
    stake_sol, hold_seconds, network_cost_sol = _strategy_parameters()
    outcomes = _build_outcomes(
        predictions,
        hold_seconds,
        force,
        output_path=REPRODUCTION_BACKTEST_OUTCOMES,
    )
    selected_entries = int(predictions.selected.sum())
    target_entries = int(predictions.label.sum())
    output: dict[str, object] = {
        "status": "PRIMARY_SOURCE_BUILT_PART3",
        "selector": "corrected frozen Part 2 relationship model",
        "row_level_inputs": "fresh predictions and gated June trades only",
        "selection_overlap": {
            "replica_entries": selected_entries,
            "target_entries": target_entries,
            "overlap": int((predictions.selected.eq(1) & predictions.label.eq(1)).sum()),
            "precision_vs_target": float(
                predictions.loc[predictions.selected.eq(1), "label"].mean()
            ),
            "recall_of_target": float(
                predictions.loc[predictions.label.eq(1), "selected"].mean()
            ),
        },
        "strategy": {
            "stake_sol": stake_sol,
            "hold_seconds": hold_seconds,
            "round_trip_network_cost_sol": network_cost_sol,
            "accounting": "network-cost-adjusted and gross of proportional Pump swap fees",
        },
        "marginal_execution": {},
        "actual_target_cashflow_june": _actual_target_june(),
    }
    for delay in (0, 1, 2):
        delayed = outcomes[outcomes.delay_slots.eq(delay)]
        replica, _ = backtest_metrics(
            delayed[delayed.selected.eq(1)],
            selected_entries,
            stake_sol,
            network_cost_sol,
        )
        target, _ = backtest_metrics(
            delayed[delayed.label.eq(1)],
            target_entries,
            stake_sol,
            network_cost_sol,
        )
        output["marginal_execution"][f"delay_{delay}"] = {
            "reproduced_selector": replica,
            "target_equal_stake": target,
        }
    if aggregate_output is not None:
        aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        aggregate_output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def run_frozen_reproduction(
    *,
    force_from_raw: bool = False,
    force_cache: bool = False,
    force_backtest: bool = False,
) -> dict[str, object]:
    """Fit and score a fresh copy of the immutable promoted Part 2 recipe."""
    ensure_output_dirs()
    REPRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    recipe = _recipe()
    feature_path, feature_state = build_reproduction_feature_cache(
        force_from_raw=force_from_raw,
        force_cache=force_cache,
    )
    started = time.monotonic()
    train = _load_reproduction_frame(feature_path, "train")
    june = _load_reproduction_frame(feature_path, "june")
    expected_state = recipe["expected"]["feature_state"]
    if len(train) != int(expected_state["sampled_training_rows"]):
        raise RuntimeError("sampled training row count changed")
    if int(train.label.sum()) != int(expected_state["training_population_positives"]):
        raise RuntimeError("sampled training positive count changed")
    bundle = _fit_frozen_model(train, recipe)
    joblib.dump(bundle.model, REPRODUCTION_MODEL)
    score = predict(bundle, june)
    threshold = float(recipe["threshold"])
    labels = june.label.to_numpy(dtype=np.uint8)
    observed_metrics = metrics(labels, score, threshold)
    expected_metrics = dict(recipe["expected"]["june_metrics"])
    tolerance = float(recipe["verification"]["metric_absolute_tolerance"])
    _assert_metrics(observed_metrics, expected_metrics, tolerance)
    fingerprints = _score_fingerprints(
        june.token_address, labels, score, threshold
    )
    expected_fingerprints = recipe["expected"]["prediction_fingerprints"]
    for key in (
        "score_quantized_11_sha256",
        "full_ranking_sha256",
        "selected_token_sha256",
        "token_label_selection_sha256",
    ):
        if fingerprints[key] != expected_fingerprints[key]:
            raise RuntimeError(f"June prediction fingerprint differs: {key}")

    predictions = pd.DataFrame(
        {
            "token_address": june.token_address,
            "block_time": june.block_time,
            "label": labels,
            "score": score,
            "selected": (score >= threshold).astype("uint8"),
        }
    )
    predictions.to_parquet(REPRODUCTION_PREDICTIONS, index=False)
    part3 = reproduce_part3_overlap_and_backtest(
        predictions, force=force_backtest
    )
    overlap = part3["selection_overlap"]
    if (
        overlap["replica_entries"] != observed_metrics["predicted_entries"]
        or overlap["overlap"] != observed_metrics["true_positives"]
    ):
        raise RuntimeError("Part 3 overlap does not reconcile to reproduced metrics")
    part3_reference = ROOT / str(recipe["primary_part3_reference"]["path"])
    if _sha256(part3_reference) != recipe["primary_part3_reference"]["sha256"]:
        raise RuntimeError("primary Part 3 aggregate reference hash changed")
    if part3 != json.loads(part3_reference.read_text()):
        raise RuntimeError("fresh primary Part 3 result differs from the frozen aggregate")

    booster = bundle.model.named_steps["model"].booster_
    report: dict[str, object] = {
        "status": "PASS_FRESH_TRAINING_AND_JUNE_SCORING",
        "recipe_sha256": _sha256(RECIPE),
        "runtime_versions": _runtime_versions(),
        "feature_state": feature_state,
        "training": {
            "window": recipe["training_window"],
            "rows_after_frozen_sampling": int(len(train)),
            "positives": int(train.label.sum()),
            "negative_sampling": recipe["negative_sampling"],
            "model_output": str(REPRODUCTION_MODEL),
            "model_output_sha256": _sha256(REPRODUCTION_MODEL),
            "lightgbm_trees": int(booster.num_trees()),
            "lightgbm_model_string_sha256": hashlib.sha256(
                booster.model_to_string().encode()
            ).hexdigest(),
        },
        "verification": {
            "model_serialization": recipe["verification"]["model_serialization"],
            "frozen_reference_model_string_sha256": recipe["expected"][
                "frozen_model_string_sha256"
            ],
            "prediction_fingerprints": fingerprints,
            "metric_absolute_tolerance": tolerance,
            "zero_selection_differences_required": True,
        },
        "june_metrics": observed_metrics,
        "june_predictions": {
            "path": str(REPRODUCTION_PREDICTIONS),
            "rows": int(len(predictions)),
            "generated_by_fresh_model": True,
        },
        "part3_reproduction": part3,
        "elapsed_seconds": time.monotonic() - started,
        "bounded_result_correction": recipe.get("bounded_correction"),
        "canonical_promoted_artifacts_written": False,
    }
    REPRODUCTION_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
