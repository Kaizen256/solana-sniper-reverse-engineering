#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from solana_sniper_reverse_engineering.config import ROOT, SUBMISSION


PUBLIC_RESOURCE_MANIFEST = SUBMISSION / "tables" / "public_resource_manifest.json"
RECIPE = SUBMISSION / "tables" / "target_relationship_reproduction_recipe.json"

PROJECT_FILES = ("pyproject.toml", "uv.lock", "README.md", "LICENSE")
RUNTIME_TABLES = (
    "curve_replay_results.json",
    "developer_sell_outcome_results.json",
    "feature_dictionary.csv",
    "historical_outcome_audit.json",
    "methodological_audit.json",
    "profitable_disagreement_results.json",
    "ranking_hard_negative_results.json",
    "robustness_summary.json",
    "target_relationship_feature_dictionary.csv",
    "target_relationship_feature_importance.csv",
    "target_relationship_primary_backtest.json",
    "target_relationship_reproduction_recipe.json",
    "target_relationship_validation.json",
    "third_pass_head_to_head.csv",
)
RUNTIME_MODULES = (
    "__init__.py",
    "backtest.py",
    "behavior.py",
    "config.py",
    "curve_replay.py",
    "feature_store.py",
    "fee_ledger.py",
    "frozen_reproduction.py",
    "message_features.py",
    "modeling.py",
    "target_relationship.py",
    "third_pass.py",
)
RUNTIME_FIGURES = (
    "05_model_diagnostics.png",
    "06_backtest_comparison.png",
)
ROW_LEVEL_SUFFIXES = (
    ".parquet",
    ".jsonl",
    ".jsonl.gz",
    ".zst",
    ".joblib",
    ".npy",
    ".npz",
)
PROHIBITED_BASENAMES = {
    "frozen_part2_reproduction_features.parquet",
    "june_curve_replay_outcomes.parquet",
    "model.joblib",
    "target_relationship_june_predictions.parquet",
    "targeted_raw_block_trade_events.csv",
    "third_pass_june_strategy_predictions.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_runtime_source(output: Path) -> None:
    for filename in RUNTIME_MODULES:
        copy_file(
            ROOT / "src" / "solana_sniper_reverse_engineering" / filename,
            output / "src" / "solana_sniper_reverse_engineering" / filename,
        )
    for filename in PROJECT_FILES:
        copy_file(ROOT / filename, output / filename)
    for filename in RUNTIME_TABLES:
        copy_file(SUBMISSION / "tables" / filename, output / "submission/tables" / filename)
    for filename in RUNTIME_FIGURES:
        copy_file(SUBMISSION / "figures" / filename, output / "submission/figures" / filename)


def _publication_violations(output: Path, files: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in files:
        relative = path.relative_to(output).as_posix()
        lower = path.name.lower()
        if (
            path.name in PROHIBITED_BASENAMES
            or any(lower.endswith(suffix) for suffix in ROW_LEVEL_SUFFIXES)
            or "public_cache" in path.parts
            or "data/raw" in relative
            or "target_relationship_rescue" in path.parts
            or path.stat().st_size > 10_000_000
        ):
            violations.append(relative)
    return violations


def build(output: Path, *, force: bool = False) -> dict[str, object]:
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _copy_runtime_source(output)

    files = sorted(path for path in output.rglob("*") if path.is_file())
    prohibited = _publication_violations(output, files)
    if prohibited:
        raise RuntimeError(f"public Resource contains prohibited files: {prohibited}")
    entries = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    recipe = json.loads(RECIPE.read_text())
    aggregate_paths = [
        f"submission/tables/{filename}" for filename in RUNTIME_TABLES
    ]
    figure_paths = [
        f"submission/figures/{filename}" for filename in RUNTIME_FIGURES
    ]
    manifest: dict[str, object] = {
        "resource_role": "optional source-only publication bundle",
        "public_resource_required_for_current_notebook": False,
        "publication_status": "READY_FOR_MANUAL_AGGREGATE_PERMISSION_REVIEW",
        "safe_to_publish_now": False,
        "publication_blockers": [
            "The competition rules do not themselves grant this automated pass authority to publish derived aggregate tables or figures; manually confirm organizer/Kaggle permission for every explicitly listed aggregate before upload.",
        ],
        "contains_competition_raw_data": False,
        "contains_row_level_competition_derivatives": False,
        "contains_derived_feature_or_label_cache": False,
        "contains_strategy_prediction_or_selection_cache": False,
        "contains_exact_curve_outcome_cache": False,
        "contains_trained_models": False,
        "contains_small_aggregate_outputs_and_figures": True,
        "aggregate_files_requiring_manual_permission_review": aggregate_paths,
        "figure_files_requiring_manual_permission_review": figure_paths,
        "source_and_configuration_files": [
            item["path"]
            for item in entries
            if not str(item["path"]).startswith("submission/")
        ],
        "competition_input_policy": "competition data is not included; the notebook resolves the documented repository data/raw layout or optional SOLANA_* environment overrides",
        "notebook_output_policy": "regenerated row-level features, labels, scores, selections, models, and replay outcomes remain local ignored working artifacts and are not included in this source bundle",
        "required_competition_inputs": recipe["authorized_inputs"],
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "prohibited_files": prohibited,
    }
    PUBLIC_RESOURCE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "resource_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the optional source-only publication bundle"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "public_notebook_resource",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build(args.output.resolve(), force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()