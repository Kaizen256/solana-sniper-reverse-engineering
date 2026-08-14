from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    ACTIVE_ERA_START,
    ARTIFACTS,
    BOUGHT_TXS,
    INTERIM,
    JULY_START,
    JUNE_START,
    MAY_START,
    NOT_BOUGHT_TXS,
    SUBMISSION,
    ensure_output_dirs,
)
from .feature_store import FEATURE_STORE
from .message_features import COMPUTE_BUDGET_PROGRAM, PUMP_PROGRAM, SYSTEM_PROGRAM, parse_create
from .modeling import ModelBundle, fit_lightgbm, metrics, predict
from .target_relationship import (
    APRIL_START,
    FREEZE_MANIFEST as TARGET_FREEZE_MANIFEST,
    TARGET_RELATIONSHIP_FULL,
    TARGET_RELATIONSHIP_PRE_JUNE,
    TARGET_SIGNER_EVENTS_FULL,
    TARGET_SIGNER_EVENTS_PRE_JUNE,
    TRAINING_NEGATIVE_STRIDE,
    _model_summary,
    _rank_summary,
    audit_relationship_features,
    build_relationship_features,
    raw_event_audit,
)
from .third_pass import DEV_SELL_FEATURES, QUALITY_FEATURES


FINGERPRINT_TYPES = (
    "alt_table",
    "system_transfer_destination",
    "extra_signer",
    "lookup_writable",
    "program_set",
    "program_sequence",
)
ACCOUNT_FINGERPRINT_TYPES = FINGERPRINT_TYPES[:4]
ARCHETYPE_FINGERPRINT_TYPES = FINGERPRINT_TYPES[4:]
FINGERPRINT_METRICS = (
    "current_count",
    "known_count",
    "prior_target_buy_max",
    "prior_target_buy_sum",
    "seconds_since_target_min",
    "prior_target_rate_max",
    "prior_occurrence_max",
)


def _feature_name(fingerprint_type: str, metric: str) -> str:
    return f"message_{fingerprint_type}_{metric}"


ACCOUNT_IDENTITY_FEATURES = [
    _feature_name(kind, metric)
    for kind in ACCOUNT_FINGERPRINT_TYPES
    for metric in FINGERPRINT_METRICS
]
ARCHETYPE_FEATURES = [
    _feature_name(kind, metric)
    for kind in ARCHETYPE_FINGERPRINT_TYPES
    for metric in FINGERPRINT_METRICS
]
ALL_MESSAGE_IDENTITY_FEATURES = ACCOUNT_IDENTITY_FEATURES + ARCHETYPE_FEATURES

FINGERPRINT_SCHEMA = pa.schema(
    [
        ("token_address", pa.string()),
        ("block_time", pa.int64()),
        ("fingerprint_type", pa.string()),
        ("fingerprint_value", pa.string()),
    ]
)

MESSAGE_IDENTITY_DIR = INTERIM / "message_identity"
PRE_JUNE_BOUGHT = MESSAGE_IDENTITY_DIR / "bought_pre_june.parquet"
PRE_JUNE_NOT_BOUGHT = MESSAGE_IDENTITY_DIR / "not_bought_pre_june.parquet"
FULL_BOUGHT = MESSAGE_IDENTITY_DIR / "bought_full.parquet"
FULL_NOT_BOUGHT = MESSAGE_IDENTITY_DIR / "not_bought_full.parquet"
PRE_JUNE_FEATURES = INTERIM / "message_identity_features_pre_june.parquet"
FULL_FEATURES = INTERIM / "message_identity_features_full.parquet"

MODEL_DIR = ARTIFACTS / "models" / "message_identity_rescue"
VALIDATION_RESULTS = SUBMISSION / "tables" / "message_identity_validation.json"
JUNE_RESULTS = SUBMISSION / "tables" / "message_identity_june_reporting.json"
FREEZE_MANIFEST = MODEL_DIR / "freeze_manifest.json"
PREDICTIONS = ARTIFACTS / "tables" / "message_identity_june_predictions.parquet"


def _sql(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _connection(memory: str = "30GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    con.execute("SET threads=20")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    temp = INTERIM / "duckdb_temp_message_identity"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{_sql(temp)}'")
    return con


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _paths(cutoff: int) -> tuple[Path, Path, Path, Path]:
    if cutoff == JUNE_START:
        return PRE_JUNE_BOUGHT, PRE_JUNE_NOT_BOUGHT, PRE_JUNE_FEATURES, TARGET_SIGNER_EVENTS_PRE_JUNE
    if cutoff == JULY_START:
        return FULL_BOUGHT, FULL_NOT_BOUGHT, FULL_FEATURES, TARGET_SIGNER_EVENTS_FULL
    raise ValueError(f"unsupported cutoff: {cutoff}")


def _create_tokens(message: dict[str, object]) -> list[str]:
    tokens: list[str] = []
    for instruction in message.get("instructions", []):  # type: ignore[union-attr]
        if instruction.get("programId") != PUMP_PROGRAM:
            continue
        accounts = instruction.get("accounts") or []
        data = instruction.get("data")
        if accounts and data and parse_create(data) is not None:
            tokens.append(str(accounts[0]))
    return list(dict.fromkeys(tokens))


def _message_fingerprints(message: dict[str, object], create_tokens: list[str]) -> list[tuple[str, str]]:
    instructions = message.get("instructions", [])  # type: ignore[assignment]
    keys = message.get("accountKeys", [])  # type: ignore[assignment]
    programs = [str(item.get("programId", "")) for item in instructions if item.get("programId")]
    fingerprints: set[tuple[str, str]] = set()
    for lookup in message.get("addressTableLookups") or []:  # type: ignore[union-attr]
        value = lookup.get("accountKey")
        if value:
            fingerprints.add(("alt_table", str(value)))
    for instruction in instructions:
        if instruction.get("programId") != SYSTEM_PROGRAM:
            continue
        parsed = instruction.get("parsed") or {}
        if parsed.get("type") == "transfer":
            destination = (parsed.get("info") or {}).get("destination")
            if destination:
                fingerprints.add(("system_transfer_destination", str(destination)))
    signers = [str(key.get("pubkey")) for key in keys if key.get("signer") and key.get("pubkey")]
    transaction_signer = signers[0] if signers else None
    excluded = set(create_tokens)
    if transaction_signer:
        excluded.add(transaction_signer)
    for signer in signers:
        if signer not in excluded:
            fingerprints.add(("extra_signer", signer))
    for key in keys:
        if key.get("source") == "lookupTable" and key.get("writable") and key.get("pubkey"):
            value = str(key["pubkey"])
            if value not in excluded:
                fingerprints.add(("lookup_writable", value))
    if programs:
        fingerprints.add(("program_set", "|".join(sorted(set(programs)))))
        fingerprints.add(("program_sequence", ">".join(programs)))
    return sorted(fingerprints)


def _extract_rows(record: dict[str, object], cutoff: int) -> list[dict[str, object]]:
    block_time = int(record["blockTime"])
    if block_time >= cutoff:
        return []
    message = record["transaction"]["message"]  # type: ignore[index]
    tokens = _create_tokens(message)
    if not tokens:
        return []
    fingerprints = _message_fingerprints(message, tokens)
    return [
        {
            "token_address": token,
            "block_time": block_time,
            "fingerprint_type": kind,
            "fingerprint_value": value,
        }
        for token in tokens
        for kind, value in fingerprints
    ]


def extract_file(source: Path, destination: Path, cutoff: int, *, force: bool = False) -> dict[str, object]:
    manifest_path = destination.with_suffix(".manifest.json")
    if destination.exists() and manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(temporary, FINGERPRINT_SCHEMA, compression="zstd", use_dictionary=True)
    batch: list[dict[str, object]] = []
    lines = in_window_lines = rows = 0
    started = time.monotonic()
    try:
        with gzip.open(source, "rb") as stream:
            for line in stream:
                lines += 1
                record = orjson.loads(line)
                if int(record["blockTime"]) < cutoff:
                    in_window_lines += 1
                extracted = _extract_rows(record, cutoff)
                batch.extend(extracted)
                rows += len(extracted)
                if len(batch) >= 100_000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=FINGERPRINT_SCHEMA))
                    batch.clear()
                if lines % 500_000 == 0:
                    elapsed = time.monotonic() - started
                    print(f"{source.name}: {lines:,} lines, {rows:,} fingerprints, {lines/elapsed:,.0f} lines/s", flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=FINGERPRINT_SCHEMA))
    finally:
        writer.close()
    os.replace(temporary, destination)
    manifest = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "cutoff_exclusive": cutoff,
        "line_count": lines,
        "in_window_line_count": in_window_lines,
        "fingerprint_rows": rows,
        "fingerprint_types": list(FINGERPRINT_TYPES),
        "strict_role": "current signed transaction.message only; transaction meta is never read",
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def extract_fingerprints(cutoff: int, *, force: bool = False) -> dict[str, object]:
    bought, not_bought, _, _ = _paths(cutoff)
    return {
        "bought": extract_file(BOUGHT_TXS, bought, cutoff, force=force),
        "not_bought": extract_file(NOT_BOUGHT_TXS, not_bought, cutoff, force=force),
    }


def build_message_identity_features(cutoff: int, *, force: bool = False) -> Path:
    bought, not_bought, output, target_events = _paths(cutoff)
    if output.exists() and not force:
        return output
    if not bought.exists() or not not_bought.exists():
        extract_fingerprints(cutoff, force=force)
    if not target_events.exists():
        raise FileNotFoundError(f"target signer events are required first: {target_events}")
    conditional_columns = []
    for kind in FINGERPRINT_TYPES:
        for metric in FINGERPRINT_METRICS:
            source = {
                "current_count": "current_count",
                "known_count": "known_count",
                "prior_target_buy_max": "prior_target_buy_max",
                "prior_target_buy_sum": "prior_target_buy_sum",
                "seconds_since_target_min": "seconds_since_target_min",
                "prior_target_rate_max": "prior_target_rate_max",
                "prior_occurrence_max": "prior_occurrence_max",
            }[metric]
            conditional_columns.append(
                f"coalesce(max(p.{source}) FILTER (WHERE p.fingerprint_type='{kind}'),0) "
                f"AS {_feature_name(kind, metric)}"
            )
    con = _connection()
    started = time.monotonic()
    con.execute(
        f"""
        COPY (
          WITH fingerprints AS (
            SELECT * FROM read_parquet('{_sql(bought)}')
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{_sql(not_bought)}')
          ), occurrence_seconds AS (
            SELECT fingerprint_type,fingerprint_value,block_time AS occurrence_time,
                   count(DISTINCT token_address)::BIGINT AS occurrences_at_second
            FROM fingerprints GROUP BY 1,2,3
          ), occurrence_state AS (
            SELECT fingerprint_type,fingerprint_value,occurrence_time,
                   sum(occurrences_at_second) OVER (
                     PARTITION BY fingerprint_type,fingerprint_value ORDER BY occurrence_time
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS cumulative_occurrences
            FROM occurrence_seconds
          ), target_seconds AS (
            SELECT f.fingerprint_type,f.fingerprint_value,e.target_buy_time,
                   count(DISTINCT f.token_address)::BIGINT AS target_buys_at_second
            FROM fingerprints f JOIN read_parquet('{_sql(target_events)}') e USING(token_address)
            GROUP BY 1,2,3
          ), target_state AS (
            SELECT fingerprint_type,fingerprint_value,target_buy_time,
                   sum(target_buys_at_second) OVER (
                     PARTITION BY fingerprint_type,fingerprint_value ORDER BY target_buy_time
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS cumulative_target_buys
            FROM target_seconds
          ), enriched AS (
            SELECT f.*,
                   coalesce(t.cumulative_target_buys,0) AS prior_target_buys,
                   coalesce(o.cumulative_occurrences,0) AS prior_occurrences,
                   t.target_buy_time AS target_state_time,
                   f.block_time-t.target_buy_time AS seconds_since_target,
                   coalesce(t.cumulative_target_buys,0)/(coalesce(o.cumulative_occurrences,0)+20.0)
                     AS target_rate_shrunk
            FROM fingerprints f
            ASOF LEFT JOIN target_state t
              ON f.fingerprint_type=t.fingerprint_type
             AND f.fingerprint_value=t.fingerprint_value
             AND f.block_time>t.target_buy_time
            ASOF LEFT JOIN occurrence_state o
              ON f.fingerprint_type=o.fingerprint_type
             AND f.fingerprint_value=o.fingerprint_value
             AND f.block_time>o.occurrence_time
          ), per_type AS (
            SELECT token_address,fingerprint_type,
                   count(*) AS current_count,
                   count(*) FILTER (WHERE prior_target_buys>0) AS known_count,
                   max(prior_target_buys) AS prior_target_buy_max,
                   sum(prior_target_buys) AS prior_target_buy_sum,
                   min(seconds_since_target) AS seconds_since_target_min,
                   max(target_rate_shrunk) AS prior_target_rate_max,
                   max(prior_occurrences) AS prior_occurrence_max,
                   max(target_state_time) AS target_state_time_max
            FROM enriched GROUP BY 1,2
          )
          SELECT d.token_address,max(p.target_state_time_max) AS message_identity_state_time,
                 {','.join(conditional_columns)}
          FROM read_parquet('{_sql(FEATURE_STORE)}') d
          LEFT JOIN per_type p USING(token_address)
          WHERE d.block_time<{cutoff}
          GROUP BY d.token_address
        ) TO '{_sql(output)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    print(f"built {output} in {time.monotonic() - started:.1f}s", flush=True)
    return output


def audit_message_identity_features(path: Path, cutoff: int) -> dict[str, int]:
    con = _connection("16GB")
    nonnegative = " OR ".join(f"m.{name}<0" for name in ALL_MESSAGE_IDENTITY_FEATURES)
    rate_invalid = " OR ".join(
        f"m.{_feature_name(kind, 'prior_target_rate_max')}<0"
        for kind in FINGERPRINT_TYPES
    )
    row = con.execute(
        f"""
        SELECT count(*) AS n_rows,count(DISTINCT m.token_address) AS tokens,
               count(*)-count(DISTINCT m.token_address) AS duplicate_tokens,
               count(*) FILTER (
                 WHERE message_identity_state_time IS NOT NULL
                   AND message_identity_state_time>=f.block_time
               ) AS future_or_equal_states,
               count(*) FILTER (WHERE {nonnegative}) AS negative_features,
               count(*) FILTER (WHERE {rate_invalid}) AS invalid_rates
        FROM read_parquet('{_sql(path)}') m
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
    if result["n_rows"] != result["expected_rows"] or any(
        result[name] for name in ("duplicate_tokens", "future_or_equal_states", "negative_features", "invalid_rates")
    ):
        raise RuntimeError(f"message identity audit failed: {result}")
    return result


def _load_frame(
    relationship_path: Path,
    message_path: Path,
    predicate: str,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    sources = {
        "f": (FEATURE_STORE, set(pq.read_schema(FEATURE_STORE).names)),
        "r": (relationship_path, set(pq.read_schema(relationship_path).names)),
        "q": (QUALITY_FEATURES, set(pq.read_schema(QUALITY_FEATURES).names)),
        "s": (DEV_SELL_FEATURES, set(pq.read_schema(DEV_SELL_FEATURES).names)),
        "m": (message_path, set(pq.read_schema(message_path).names)),
    }
    selections = []
    for name in numeric + categorical:
        matches = [alias for alias, (_, names) in sources.items() if name in names]
        if not matches:
            raise KeyError(f"feature unavailable: {name}")
        selections.append(f'{matches[0]}."{name}"')
    con = _connection()
    return con.execute(
        f"""
        SELECT f.token_address,f.tx_hash,f.tx_signer,f.block_time,f.label,{','.join(selections)}
        FROM read_parquet('{_sql(FEATURE_STORE)}') f
        JOIN read_parquet('{_sql(relationship_path)}') r USING(token_address)
        JOIN read_parquet('{_sql(QUALITY_FEATURES)}') q USING(token_address)
        JOIN read_parquet('{_sql(DEV_SELL_FEATURES)}') s USING(token_address)
        JOIN read_parquet('{_sql(message_path)}') m USING(token_address)
        WHERE {predicate}
        ORDER BY f.block_time,f.token_address
        """
    ).fetch_df()


def _window(
    relationship_path: Path,
    message_path: Path,
    base_numeric: list[str],
    categorical: list[str],
    train_end: int,
    validation_start: int,
    validation_end: int,
) -> tuple[dict[str, object], dict[str, ModelBundle], dict[str, float]]:
    maximum = base_numeric + ALL_MESSAGE_IDENTITY_FEATURES
    train = _load_frame(
        relationship_path,
        message_path,
        f"f.block_time>={ACTIVE_ERA_START} AND f.block_time<{train_end} "
        f"AND (f.label=1 OR hash(f.token_address)%{TRAINING_NEGATIVE_STRIDE}=0)",
        maximum,
        categorical,
    )
    validation = _load_frame(
        relationship_path,
        message_path,
        f"f.block_time>={validation_start} AND f.block_time<{validation_end}",
        maximum,
        categorical,
    )
    feature_sets = {
        "frozen_target_signer_baseline": base_numeric,
        "baseline_plus_account_identities": base_numeric + ACCOUNT_IDENTITY_FEATURES,
        "baseline_plus_archetypes": base_numeric + ARCHETYPE_FEATURES,
        "baseline_plus_all_message_identities": maximum,
    }
    models: dict[str, ModelBundle] = {}
    thresholds: dict[str, float] = {}
    output: dict[str, object] = {
        "train_rows": int(len(train)),
        "train_positives": int(train.label.sum()),
        "validation_rows": int(len(validation)),
        "validation_positives": int(validation.label.sum()),
    }
    baseline_ap = 0.0
    for name, features in feature_sets.items():
        model = fit_lightgbm(train, features, categorical, TRAINING_NEGATIVE_STRIDE)
        summary, _, threshold = _model_summary(model, validation)
        models[name] = model
        thresholds[name] = threshold
        output[name] = summary
        if name == "frozen_target_signer_baseline":
            baseline_ap = float(summary["pr_auc"])
    for name in feature_sets:
        if name != "frozen_target_signer_baseline":
            output[f"{name}_pr_auc_delta"] = float(output[name]["pr_auc"] - baseline_ap)
    return output, models, thresholds


def run_validation(*, force_extract: bool = False, force_features: bool = False) -> dict[str, object]:
    if not TARGET_FREEZE_MANIFEST.exists():
        raise RuntimeError("freeze the target-signer system before testing message identities")
    target_freeze = json.loads(TARGET_FREEZE_MANIFEST.read_text())
    if target_freeze.get("status") != "FROZEN_PRE_JUNE":
        raise RuntimeError(f"invalid target-signer freeze: {target_freeze}")
    ensure_output_dirs()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    extraction = extract_fingerprints(JUNE_START, force=force_extract)
    message_path = build_message_identity_features(JUNE_START, force=force_features)
    temporal_audit = audit_message_identity_features(message_path, JUNE_START)
    base_numeric = list(target_freeze["numeric_features"])
    categorical = list(target_freeze["categorical_features"])
    windows: dict[str, object] = {}
    fitted: dict[str, dict[str, ModelBundle]] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for period, train_end, start, end in (
        ("april", APRIL_START, APRIL_START, MAY_START),
        ("may", MAY_START, MAY_START, JUNE_START),
    ):
        windows[period], fitted[period], thresholds[period] = _window(
            TARGET_RELATIONSHIP_PRE_JUNE,
            message_path,
            base_numeric,
            categorical,
            train_end,
            start,
            end,
        )
        print(period, json.dumps(windows[period], indent=2, sort_keys=True), flush=True)
    candidates = (
        "baseline_plus_account_identities",
        "baseline_plus_archetypes",
        "baseline_plus_all_message_identities",
    )
    eligible = [
        name
        for name in candidates
        if all(float(windows[period][f"{name}_pr_auc_delta"]) >= 0.01 for period in ("april", "may"))
    ]
    selected = max(
        eligible,
        key=lambda name: sum(float(windows[p][name]["pr_auc"]) for p in ("april", "may")),
        default=None,
    )
    output = {
        "status": "PROMOTE" if selected else "REJECT",
        "hypothesis": "Recurring signed-message account identities and construction archetypes add material signal beyond the frozen target-signer system.",
        "decision_clock": "Current fingerprints come from transaction.message; target-conditioned fingerprint state uses only target_buy_time < candidate block_time.",
        "selection_rule": "Promote a predeclared family only for >=0.01 AP lift over the frozen target-signer baseline in both April and May.",
        "selected_model": selected,
        "extraction": extraction,
        "temporal_audit": temporal_audit,
        "features": {
            "account_identities": ACCOUNT_IDENTITY_FEATURES,
            "archetypes": ARCHETYPE_FEATURES,
        },
        "windows": windows,
        "june_opened": False,
    }
    VALIDATION_RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if selected:
        model = fitted["may"][selected]
        joblib.dump(model.model, MODEL_DIR / "model.joblib")
        selected_message_features = {
            "baseline_plus_account_identities": ACCOUNT_IDENTITY_FEATURES,
            "baseline_plus_archetypes": ARCHETYPE_FEATURES,
            "baseline_plus_all_message_identities": ALL_MESSAGE_IDENTITY_FEATURES,
        }[selected]
        freeze = {
            "status": "FROZEN_PRE_JUNE",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "module_sha256": _module_sha256(),
            "validation_results_sha256": hashlib.sha256(VALIDATION_RESULTS.read_bytes()).hexdigest(),
            "target_freeze_sha256": hashlib.sha256(TARGET_FREEZE_MANIFEST.read_bytes()).hexdigest(),
            "selected_model": selected,
            "selected_message_features": selected_message_features,
            "numeric_features": model.numeric_features,
            "categorical_features": model.categorical_features,
            "threshold_selected_on_may": thresholds["may"][selected],
            "june_scored": False,
        }
        FREEZE_MANIFEST.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def run_june_reporting(*, force_extract: bool = False, force_features: bool = False) -> dict[str, object]:
    if not FREEZE_MANIFEST.exists() or not VALIDATION_RESULTS.exists():
        raise RuntimeError("message identity family was not promoted pre-June")
    if JUNE_RESULTS.exists():
        raise RuntimeError(f"June has already been scored: {JUNE_RESULTS}")
    freeze = json.loads(FREEZE_MANIFEST.read_text())
    if freeze["module_sha256"] != _module_sha256():
        raise RuntimeError("message_identity.py changed after the pre-June freeze")
    if freeze["validation_results_sha256"] != hashlib.sha256(VALIDATION_RESULTS.read_bytes()).hexdigest():
        raise RuntimeError("message identity validation results changed after freeze")
    if freeze["target_freeze_sha256"] != hashlib.sha256(TARGET_FREEZE_MANIFEST.read_bytes()).hexdigest():
        raise RuntimeError("target-signer freeze changed after message-family selection")
    relationship_path = build_relationship_features(JULY_START, force=force_features)
    extraction = extract_fingerprints(JULY_START, force=force_extract)
    message_path = build_message_identity_features(JULY_START, force=force_features)
    target_audit = audit_relationship_features(relationship_path, JULY_START, 0)
    target_event_audit = raw_event_audit(JULY_START)
    message_audit = audit_message_identity_features(message_path, JULY_START)
    numeric = list(freeze["numeric_features"])
    categorical = list(freeze["categorical_features"])
    june = _load_frame(
        relationship_path,
        message_path,
        f"f.block_time>={JUNE_START} AND f.block_time<{JULY_START}",
        numeric,
        categorical,
    )
    pipeline = joblib.load(MODEL_DIR / "model.joblib")
    bundle = ModelBundle(pipeline, numeric, categorical, [], str(freeze["selected_model"]))
    score = predict(bundle, june)
    threshold = float(freeze["threshold_selected_on_may"])
    y = june.label.to_numpy(dtype=np.uint8)
    output = {
        "status": "FROZEN_JUNE_REPORT",
        "selected_model": freeze["selected_model"],
        "threshold": threshold,
        "metrics": {**metrics(y, score, threshold), "top_k": _rank_summary(y, score)},
        "extraction": extraction,
        "target_event_audit": target_event_audit,
        "target_temporal_audit": target_audit,
        "message_temporal_audit": message_audit,
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
    parser = argparse.ArgumentParser(description="Signed-message identity rescue experiment")
    parser.add_argument("stage", choices=("validate", "june"))
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    if args.stage == "validate":
        run_validation(force_extract=args.force_extract, force_features=args.force_features)
    else:
        run_june_reporting(force_extract=args.force_extract, force_features=args.force_features)


if __name__ == "__main__":
    main()
