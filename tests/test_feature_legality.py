from __future__ import annotations

from solana_sniper_reverse_engineering.modeling import CATEGORICAL_FEATURES, FEATURE_GROUPS


def test_model_feature_allowlist_excludes_outcome_and_identity_fields() -> None:
    features = set(CATEGORICAL_FEATURES)
    for group in FEATURE_GROUPS.values():
        features.update(group)
    prohibited = {
        "label",
        "token_address",
        "tx_hash",
        "tx_signer",
        "creator_address",
        "price_usd",
        "pnl",
        "target_wallet",
        "jito",
        "candle",
        "transaction_index",
        "block_slot",
    }
    assert not (features & prohibited)


def test_activity_features_are_explicitly_historical() -> None:
    activity_features = FEATURE_GROUPS["activity_history"]
    assert activity_features
    assert all(
        feature.startswith("hist_")
        or feature in {"history_missing", "observed_wallet_age_seconds", "seconds_since_activity"}
        for feature in activity_features
    )
