from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ACTIVE_ERA_START, ARTIFACTS, SUBMISSION, ensure_output_dirs
from .feature_store import FEATURE_STORE
from .modeling import (
    CATEGORICAL_FEATURES,
    PREDICTIONS,
    _available_features,
    _permutation_importance,
    _thresholds,
    fit_lightgbm,
    metrics,
    predict,
)


APRIL_START = 1_775_001_600
MAY_START = 1_777_593_600
SUMMARY = SUBMISSION / "tables" / "robustness_summary.json"


def _custom_frame(
    path: Path,
    numeric: list[str],
    categorical: list[str],
    predicate: str,
) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(["token_address", "tx_hash", "tx_signer", "block_time", "label"] + numeric + categorical)
    )
    select = ", ".join(f'"{column}"' for column in columns)
    con = duckdb.connect()
    con.execute("SET memory_limit='30GB'")
    con.execute("SET threads=16")
    return con.execute(
        f"SELECT {select} FROM read_parquet(?) WHERE {predicate}", [str(path)]
    ).fetch_df()


def _threshold_sensitivity(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for split, frame in predictions.groupby("split"):
        y = frame.label.to_numpy(dtype=np.uint8)
        score = frame.score.to_numpy()
        for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            item = metrics(y, score, threshold * multiplier)
            rows.append({"split": split, "threshold_multiplier": multiplier, **item})
    return pd.DataFrame(rows)


def _subgroups(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("predictions", predictions)
    joined = con.execute(
        f"""
        SELECT p.*, f.tx_signer, f.history_missing, f.message_missing,
               f.dev_buy_sol, f.prior_deploy_count
        FROM predictions p
        JOIN read_parquet('{FEATURE_STORE.as_posix()}') f USING (token_address)
        """
    ).fetch_df()
    joined["history_group"] = np.where(joined.history_missing.eq(1), "missing", "present")
    joined["dev_buy_group"] = pd.cut(
        joined.dev_buy_sol,
        bins=[-0.001, 0, 1, 3, 10, np.inf],
        labels=["zero", "(0,1]", "(1,3]", "(3,10]", ">10"],
        include_lowest=True,
    ).astype(str)
    rows: list[dict[str, object]] = []
    for split, split_frame in joined.groupby("split"):
        for dimension in ("history_group", "dev_buy_group"):
            for group, frame in split_frame.groupby(dimension):
                y = frame.label.to_numpy(dtype=np.uint8)
                if y.sum() == 0:
                    continue
                item = metrics(y, frame.score.to_numpy(), threshold)
                rows.append({"split": split, "dimension": dimension, "group": str(group), **item})
    subgroup_frame = pd.DataFrame(rows)
    subgroup_frame.to_csv(SUBMISSION / "tables" / "robustness_subgroups.csv", index=False)

    concentration_rows = []
    for split, frame in joined[joined.selected == 1].groupby("split"):
        by_signer = frame.groupby("tx_signer").agg(entries=("label", "size"), true_positives=("label", "sum"))
        ordered = by_signer.sort_values("entries", ascending=False)
        concentration_rows.append(
            {
                "split": split,
                "selected_entries": int(len(frame)),
                "selected_signers": int(len(by_signer)),
                "top1_entry_share": float(ordered.entries.head(1).sum() / len(frame)),
                "top10_entry_share": float(ordered.entries.head(10).sum() / len(frame)),
            }
        )
    pd.DataFrame(concentration_rows).to_csv(
        SUBMISSION / "tables" / "deployer_concentration.csv", index=False
    )
    return subgroup_frame


def _calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in predictions.groupby("split"):
        bins = pd.qcut(frame.score, q=20, duplicates="drop")
        grouped = (
            frame.assign(score_bin=bins.astype(str))
            .groupby("score_bin", observed=True)
            .agg(n=("label", "size"), positives=("label", "sum"), actual_rate=("label", "mean"), mean_score=("score", "mean"))
            .reset_index()
        )
        grouped.insert(0, "split", split)
        rows.append(grouped)
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(SUBMISSION / "tables" / "calibration_bins.csv", index=False)
    return output


def run() -> dict[str, object]:
    ensure_output_dirs()
    predictions = pd.read_parquet(PREDICTIONS)
    operating = json.loads((ARTIFACTS / "models" / "operating_point.json").read_text())
    threshold = float(operating["threshold"])
    sensitivity = _threshold_sensitivity(predictions, threshold)
    sensitivity.to_csv(SUBMISSION / "tables" / "threshold_sensitivity.csv", index=False)
    subgroups = _subgroups(predictions, threshold)
    calibration = _calibration(predictions)

    numeric, categorical = _available_features(FEATURE_STORE)
    stride = 2
    early_train = _custom_frame(
        FEATURE_STORE,
        numeric,
        categorical,
        f"block_time >= {ACTIVE_ERA_START} AND block_time < {APRIL_START} "
        f"AND (label=1 OR hash(token_address)%{stride}=0)",
    )
    april = _custom_frame(
        FEATURE_STORE,
        numeric,
        categorical,
        f"block_time >= {APRIL_START} AND block_time < {MAY_START}",
    )
    april_model = fit_lightgbm(early_train, numeric, categorical, stride)
    april_score = predict(april_model, april)
    april_threshold = _thresholds(april.label.to_numpy(), april_score)["max_f1"]
    april_metrics = metrics(april.label.to_numpy(), april_score, april_threshold)
    april_importance = _permutation_importance(
        april_model, april, np.random.default_rng(20260811), sample_size=100_000
    )
    april_importance.to_csv(SUBMISSION / "tables" / "feature_importance_april.csv", index=False)
    may_importance = pd.read_csv(SUBMISSION / "tables" / "feature_importance.csv")
    april_top = list(april_importance.feature.head(10))
    may_top = list(may_importance.feature.head(10))
    overlap = sorted(set(april_top) & set(may_top))

    output: dict[str, object] = {
        "alternate_temporal_validation": {
            "train": "2026-03-12 through 2026-03-31, all positives + deterministic 1/2 negatives",
            "validation": "2026-04-01 through 2026-04-30, full population",
            "metrics": april_metrics,
        },
        "top10_stability": {
            "april_top10": april_top,
            "may_top10": may_top,
            "intersection": overlap,
            "intersection_count": len(overlap),
            "jaccard": len(overlap) / len(set(april_top) | set(may_top)),
        },
        "threshold_sensitivity_path": "submission/tables/threshold_sensitivity.csv",
        "subgroup_path": "submission/tables/robustness_subgroups.csv",
        "calibration_path": "submission/tables/calibration_bins.csv",
        "deployer_concentration_path": "submission/tables/deployer_concentration.csv",
        "notes": [
            "Final model selection was based on May only; June threshold rows are robustness reporting, not retuning.",
            "Observed wallet age/counts are left-censored for wallets hitting the provider's 10,000-event cap.",
        ],
    }
    SUMMARY.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Falsify temporal and subgroup model conclusions")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
