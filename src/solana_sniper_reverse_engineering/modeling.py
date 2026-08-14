from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb
import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

from .config import (
    ACTIVE_ERA_START,
    ARTIFACTS,
    JUNE_START,
    MAY_START,
    PROCESSED,
    SUBMISSION,
    ensure_output_dirs,
)
from .feature_store import FEATURE_STORE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_DIR = ARTIFACTS / "models"
PREDICTIONS = ARTIFACTS / "tables" / "deployment_predictions.parquet"
METRICS = SUBMISSION / "tables" / "classification_metrics.json"

CATEGORICAL_FEATURES = ["uri_provider", "dev_buy_kind", "version"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "timing": ["deploy_hour_utc", "deploy_day_of_week_utc", "days_since_2026_start"],
    "deployment_history": [
        "deployments_at_second",
        "prior_deploy_count",
        "prior_deploy_count_1h",
        "prior_deploy_count_1d",
        "prior_deploy_count_7d",
        "prior_deploy_count_30d",
        "seconds_since_prior_deploy",
    ],
    "metadata": [
        "name_length",
        "symbol_length",
        "uri_length",
        "name_word_count",
        "name_digit_count",
        "symbol_digit_count",
        "name_non_ascii_count",
        "symbol_non_ascii_count",
        "name_upper_fraction",
        "symbol_upper_fraction",
        "name_symbol_same_normalized",
        "name_has_url",
        "name_has_dollar",
        "name_has_emoji_or_non_ascii",
        "symbol_has_emoji_or_non_ascii",
        "prior_name_count",
        "seconds_since_prior_name",
        "prior_symbol_count",
        "seconds_since_prior_symbol",
        "signer_prior_name_count",
        "signer_prior_symbol_count",
        "message_missing",
    ],
    "signed_message": [
        "n_message_instructions",
        "n_account_keys",
        "n_signers",
        "n_writable_accounts",
        "n_address_table_lookups",
        "n_compute_budget_instructions",
        "compute_unit_limit",
        "compute_unit_price_micro_lamports",
        "n_system_transfers",
        "system_transfer_sol",
        "max_system_transfer_sol",
        "n_pump_instructions",
        "n_create_instructions",
        "create_instruction_index",
        "has_dev_buy",
        "dev_buy_sol",
        "dev_buy_over_1000_sol",
        "dev_buy_instruction_index",
    ],
    "activity_history": [
        "history_missing",
        "observed_wallet_age_seconds",
        "seconds_since_activity",
        "hist_event_count",
        "hist_tx_count",
        "hist_buy_count",
        "hist_sell_count",
        "hist_launch_count",
        "hist_burn_count",
        "hist_open_close_count",
        "hist_pump_event_count",
        "hist_cost_usd_sum",
        "hist_quote_sol_sum",
        "hist_gas_native_sum",
        "hist_priority_fee_sum",
        "hist_tip_fee_sum",
    ]
    + [
        f"{name}_{window}"
        for name in (
            "hist_event_count",
            "hist_tx_count",
            "hist_buy_count",
            "hist_sell_count",
            "hist_launch_count",
            "hist_burn_count",
            "hist_open_close_count",
            "hist_pump_event_count",
        )
        for window in ("1d", "7d", "30d")
    ],
}


@dataclass
class ModelBundle:
    model: object
    numeric_features: list[str]
    categorical_features: list[str]
    feature_names_out: list[str]
    name: str


def _available_features(path: Path, groups: list[str] | None = None) -> tuple[list[str], list[str]]:
    available = set(pq.read_schema(path).names)
    selected_groups = groups or list(FEATURE_GROUPS)
    numeric = [
        feature
        for group in selected_groups
        for feature in FEATURE_GROUPS[group]
        if feature in available
    ]
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in available]
    return list(dict.fromkeys(numeric)), categorical


def _load_split(
    path: Path,
    numeric: list[str],
    categorical: list[str],
    split: str,
    training_negative_stride: int = 10,
) -> pd.DataFrame:
    if split == "train":
        predicate = (
            f"block_time >= {ACTIVE_ERA_START} AND block_time < {MAY_START} AND "
            f"(label=1 OR hash(token_address) % {training_negative_stride}=0)"
        )
    elif split == "validation":
        predicate = f"block_time >= {MAY_START} AND block_time < {JUNE_START}"
    elif split == "test":
        predicate = f"block_time >= {JUNE_START}"
    elif split == "april_validation":
        predicate = "block_time >= 1775001600 AND block_time < 1777593600"
    else:
        raise ValueError(split)
    columns = list(dict.fromkeys(["token_address", "tx_hash", "block_time", "label"] + numeric + categorical))
    query_columns = ", ".join(f'"{column}"' for column in columns)
    con = duckdb.connect()
    con.execute("SET memory_limit='24GB'")
    con.execute("SET threads=16")
    return con.execute(
        f"SELECT {query_columns} FROM read_parquet(?) WHERE {predicate} "
        "ORDER BY block_time, token_address",
        [str(path)],
    ).fetch_df()


def _preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_logistic(
    train: pd.DataFrame, numeric: list[str], categorical: list[str], stride: int
) -> ModelBundle:
    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale=True)),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    max_iter=250,
                    solver="lbfgs",
                    n_jobs=16,
                    random_state=20260811,
                ),
            ),
        ]
    )
    weights = np.where(train.label.to_numpy() == 1, 1.0, float(stride))
    pipeline.fit(train[numeric + categorical], train.label, model__sample_weight=weights)
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    return ModelBundle(pipeline, numeric, categorical, names, "regularized_logistic")


def fit_hist_gradient_boosting(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    stride: int,
    name: str = "constrained_hist_gradient_boosting",
) -> ModelBundle:
    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale=False)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=180,
                    max_leaf_nodes=15,
                    max_depth=5,
                    min_samples_leaf=80,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=15,
                    random_state=20260811,
                ),
            ),
        ]
    )
    weights = np.where(train.label.to_numpy() == 1, 1.0, float(stride))
    pipeline.fit(train[numeric + categorical], train.label, model__sample_weight=weights)
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    return ModelBundle(pipeline, numeric, categorical, names, name)


def fit_lightgbm(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    stride: int,
    extra_sample_weight: np.ndarray | None = None,
) -> ModelBundle:
    """A stronger but still shallow/regularized tree ensemble for validation."""
    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale=False)),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=500,
                    learning_rate=0.045,
                    num_leaves=31,
                    max_depth=7,
                    min_child_samples=120,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.2,
                    reg_lambda=2.0,
                    max_bin=127,
                    n_jobs=20,
                    random_state=20260811,
                    verbosity=-1,
                ),
            ),
        ]
    )
    weights = np.where(train.label.to_numpy() == 1, 1.0, float(stride))
    if extra_sample_weight is not None:
        if len(extra_sample_weight) != len(train):
            raise ValueError("extra_sample_weight must align one-to-one with train")
        weights = weights * np.asarray(extra_sample_weight, dtype=float)
    pipeline.fit(train[numeric + categorical], train.label, model__sample_weight=weights)
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    return ModelBundle(pipeline, numeric, categorical, names, "regularized_lightgbm")


def predict(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    inputs = frame[bundle.numeric_features + bundle.categorical_features]
    if hasattr(bundle.model, "predict_proba"):
        return bundle.model.predict_proba(inputs)[:, 1]
    return np.asarray(bundle.model.predict(inputs), dtype=float)


def _thresholds(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = int(np.nanargmax(f1))
    result = {"max_f1": float(thresholds[best])}
    for target in (0.10, 0.20, 0.30, 0.50):
        eligible = np.flatnonzero(precision[:-1] >= target)
        if eligible.size:
            index = eligible[np.argmax(recall[:-1][eligible])]
            result[f"precision_{target:.2f}"] = float(thresholds[index])
    return result


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    selected = score >= threshold
    positives = int(y.sum())
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, score))
    precision = float(precision_score(y, selected, zero_division=0))
    return {
        "rows": int(y.size),
        "positives": positives,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lift_over_prevalence": float(pr_auc / prevalence) if prevalence else 0.0,
        "roc_auc": float(roc_auc_score(y, score)),
        "threshold": float(threshold),
        "precision": precision,
        "precision_lift_over_prevalence": float(precision / prevalence) if prevalence else 0.0,
        "recall": float(recall_score(y, selected, zero_division=0)),
        "f1": float(f1_score(y, selected, zero_division=0)),
        "predicted_entries": int(selected.sum()),
        "predicted_entry_rate": float(selected.mean()),
        "true_positives": int((selected & (y == 1)).sum()),
    }


def _permutation_importance(
    bundle: ModelBundle,
    validation: pd.DataFrame,
    rng: np.random.Generator,
    sample_size: int = 200_000,
) -> pd.DataFrame:
    if len(validation) > sample_size:
        indices = rng.choice(len(validation), size=sample_size, replace=False)
        sample = validation.iloc[indices].copy()
    else:
        sample = validation.copy()
    y = sample.label.to_numpy()
    baseline = average_precision_score(y, predict(bundle, sample))
    rows = []
    for feature in bundle.numeric_features + bundle.categorical_features:
        original = sample[feature].copy()
        sample[feature] = rng.permutation(original.to_numpy())
        permuted = average_precision_score(y, predict(bundle, sample))
        sample[feature] = original
        rows.append(
            {
                "feature": feature,
                "baseline_pr_auc": baseline,
                "permuted_pr_auc": permuted,
                "importance_pr_auc_drop": baseline - permuted,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_pr_auc_drop", ascending=False)


def _feature_effects(
    bundle: ModelBundle,
    validation: pd.DataFrame,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    sample = validation.sample(min(300_000, len(validation)), random_state=20260811).copy()
    sample["prediction"] = predict(bundle, sample)
    rows: list[dict[str, object]] = []
    for rank, feature in enumerate(importance.feature.head(10), start=1):
        ordered_numeric = feature not in bundle.categorical_features and sample[feature].nunique(dropna=True) > 12
        if not ordered_numeric:
            bins = sample[feature].fillna("<missing>").astype(str)
        else:
            try:
                bins = pd.qcut(sample[feature], q=10, duplicates="drop")
            except (ValueError, TypeError):
                bins = sample[feature].fillna(-1).astype(str)
                ordered_numeric = False
        grouped = (
            sample.assign(effect_bin=bins)
            .groupby("effect_bin", observed=True, sort=True)
            .agg(n=("label", "size"), actual_buy_rate=("label", "mean"), mean_prediction=("prediction", "mean"))
            .reset_index()
        )
        correlation = (
            spearmanr(np.arange(len(grouped)), grouped.mean_prediction).statistic
            if ordered_numeric and len(grouped) > 1
            else np.nan
        )
        direction = "increasing" if correlation > 0.3 else "decreasing" if correlation < -0.3 else "non-monotonic/mixed"
        for item in grouped.to_dict("records"):
            item["effect_bin"] = str(item["effect_bin"])
            rows.append({"rank": rank, "feature": feature, "direction": direction, **item})
    return pd.DataFrame(rows)


def _surrogate(bundle: ModelBundle, train: pd.DataFrame, threshold: float) -> str:
    sample = train.sample(min(250_000, len(train)), random_state=20260811)
    transformed = bundle.model.named_steps["preprocess"].transform(
        sample[bundle.numeric_features + bundle.categorical_features]
    )
    target = predict(bundle, sample)
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=500, random_state=20260811)
    weights = np.where(target >= threshold, 10.0, 1.0)
    tree.fit(transformed, target, sample_weight=weights)
    return export_text(tree, feature_names=bundle.feature_names_out, decimals=4)


def _plot_diagnostics(
    validation_y: np.ndarray,
    validation_score: np.ndarray,
    threshold: float,
    importance: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    precision, recall, _ = precision_recall_curve(validation_y, validation_score)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(recall, precision, color="#176B87")
    chosen = metrics(validation_y, validation_score, threshold)
    axes[0].scatter([chosen["recall"]], [chosen["precision"]], color="#E07A2D", zorder=3)
    axes[0].axhline(validation_y.mean(), linestyle="--", color="grey", label="prevalence")
    axes[0].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Preserved control: May precision–recall",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend()
    top = importance.head(10).sort_values("importance_pr_auc_drop")
    axes[1].barh(top.feature, top.importance_pr_auc_drop, color="#176B87")
    axes[1].set(
        xlabel="Permutation PR-AUC drop", title="Preserved-control permutation signals"
    )
    fig.tight_layout()
    for directory in (ARTIFACTS / "figures", SUBMISSION / "figures"):
        fig.savefig(directory / "05_model_diagnostics.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(path: Path = FEATURE_STORE, force: bool = False) -> dict[str, object]:
    del force
    ensure_output_dirs()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stride = 10
    numeric, categorical = _available_features(path)
    train = _load_split(path, numeric, categorical, "train", stride)
    validation = _load_split(path, numeric, categorical, "validation", stride)
    test = _load_split(path, numeric, categorical, "test", stride)
    y_val = validation.label.to_numpy(dtype=np.uint8)

    experiments: dict[str, dict[str, float | int]] = {}
    baseline_score = np.full(len(validation), train.label.mean())
    experiments["prevalence_baseline"] = metrics(y_val, baseline_score, 1.0)

    timing_numeric, timing_cat = _available_features(path, ["timing"])
    time_model = fit_hist_gradient_boosting(train, timing_numeric, [], stride, "time_only_hgb")
    time_score = predict(time_model, validation)
    time_threshold = _thresholds(y_val, time_score)["max_f1"]
    experiments[time_model.name] = metrics(y_val, time_score, time_threshold)

    logistic = fit_logistic(train, numeric, categorical, stride)
    logistic_score = predict(logistic, validation)
    logistic_threshold = _thresholds(y_val, logistic_score)["max_f1"]
    experiments[logistic.name] = metrics(y_val, logistic_score, logistic_threshold)

    no_activity_groups = [group for group in FEATURE_GROUPS if group != "activity_history"]
    base_numeric, base_cat = _available_features(path, no_activity_groups)
    base_hgb = fit_hist_gradient_boosting(train, base_numeric, base_cat, stride, "hgb_without_activity")
    base_score = predict(base_hgb, validation)
    base_threshold = _thresholds(y_val, base_score)["max_f1"]
    experiments[base_hgb.name] = metrics(y_val, base_score, base_threshold)

    full_hgb = fit_hist_gradient_boosting(train, numeric, categorical, stride)
    full_score = predict(full_hgb, validation)
    thresholds = _thresholds(y_val, full_score)
    experiments[full_hgb.name] = metrics(y_val, full_score, thresholds["max_f1"])

    lightgbm_stride = 2
    lightgbm_train = _load_split(path, numeric, categorical, "train", lightgbm_stride)
    lightgbm = fit_lightgbm(lightgbm_train, numeric, categorical, lightgbm_stride)
    lightgbm_score = predict(lightgbm, validation)
    lightgbm_threshold = _thresholds(y_val, lightgbm_score)["max_f1"]
    experiments[lightgbm.name] = metrics(y_val, lightgbm_score, lightgbm_threshold)

    candidates = [
        (logistic, logistic_score),
        (base_hgb, base_score),
        (full_hgb, full_score),
        (lightgbm, lightgbm_score),
    ]
    chosen, validation_score = max(
        candidates, key=lambda item: average_precision_score(y_val, item[1])
    )
    chosen_thresholds = _thresholds(y_val, validation_score)
    chosen_threshold = chosen_thresholds["max_f1"]
    joblib.dump(chosen.model, MODEL_DIR / "final_model.joblib")
    (MODEL_DIR / "model_features.json").write_text(
        json.dumps(
            {
                "numeric_features": chosen.numeric_features,
                "categorical_features": chosen.categorical_features,
                "transformed_feature_names": chosen.feature_names_out,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (MODEL_DIR / "operating_point.json").write_text(
        json.dumps(
            {
                "model": chosen.name,
                "selection_policy": "fixed score threshold maximizing F1 on May validation",
                "threshold": chosen_threshold,
                "candidate_thresholds": chosen_thresholds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # June is reporting-only here. The final threshold is fixed from May above;
    # an earlier HGB run had already inspected June, as disclosed in project docs.
    test_score = predict(chosen, test)
    y_test = test.label.to_numpy(dtype=np.uint8)
    final_test_metrics = metrics(y_test, test_score, chosen_threshold)
    validation_metrics = metrics(y_val, validation_score, chosen_threshold)

    prediction_frames = []
    for split_name, frame, score in (
        ("validation", validation, validation_score),
        ("test", test, test_score),
    ):
        prediction_frames.append(
            pd.DataFrame(
                {
                    "token_address": frame.token_address,
                    "tx_hash": frame.tx_hash,
                    "block_time": frame.block_time,
                    "label": frame.label.astype("uint8"),
                    "split": split_name,
                    "score": score,
                    "selected": (score >= chosen_threshold).astype("uint8"),
                }
            )
        )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(PREDICTIONS, index=False)

    rng = np.random.default_rng(20260811)
    importance = _permutation_importance(chosen, validation, rng)
    importance.to_csv(SUBMISSION / "tables" / "feature_importance.csv", index=False)
    effects = _feature_effects(chosen, validation, importance)
    effects.to_csv(SUBMISSION / "tables" / "feature_effects.csv", index=False)
    surrogate_text = _surrogate(chosen, train, chosen_threshold)
    (SUBMISSION / "tables" / "surrogate_rules.txt").write_text(surrogate_text)
    _plot_diagnostics(y_val, validation_score, chosen_threshold, importance)

    top_features = []
    for rank, row in enumerate(importance.head(10).itertuples(index=False), start=1):
        feature_effect = effects[effects.feature == row.feature]
        top_features.append(
            {
                "rank": rank,
                "feature": row.feature,
                "permutation_pr_auc_drop": float(row.importance_pr_auc_drop),
                "direction": feature_effect.direction.iloc[0] if not feature_effect.empty else "unresolved",
            }
        )
    output: dict[str, object] = {
        "validation_policy": {
            "train": "2026-03-12 through 2026-04-30 (target active era); chosen LightGBM uses all positives and deterministic 1/2 negatives with inverse sampling weights (baselines use 1/10)",
            "validation": "2026-05-01 through 2026-05-31; full population",
            "final_test": "2026-06-01 through 2026-06-30; full reporting population. LightGBM and its threshold were selected on May only, but an earlier HGB run had already inspected June.",
        },
        "experiments_may": experiments,
        "chosen_model": chosen.name,
        "chosen_threshold": chosen_threshold,
        "validation": validation_metrics,
        "final_test": final_test_metrics,
        "top_features": top_features,
        "surrogate_rules_path": "submission/tables/surrogate_rules.txt",
    }
    METRICS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train chronological interpretable bot-selection models")
    parser.add_argument("--features", type=Path, default=FEATURE_STORE)
    args = parser.parse_args()
    run(args.features)


if __name__ == "__main__":
    main()
