#!/usr/bin/env python3
"""Run the complete competition pipeline from immutable local raw data."""

from solana_sniper_reverse_engineering.backtest import run as run_backtest
from solana_sniper_reverse_engineering.behavior import run as run_behavior
from solana_sniper_reverse_engineering.feature_store import run as run_features
from solana_sniper_reverse_engineering.message_features import run as run_messages
from solana_sniper_reverse_engineering.methodological_audit import run as run_audit
from solana_sniper_reverse_engineering.modeling import run as run_modeling
from solana_sniper_reverse_engineering.robustness import run as run_robustness


def main() -> None:
    run_behavior()
    run_messages()
    features = run_features()
    run_modeling(features)
    run_backtest(force=True)
    run_robustness()
    run_audit(features)


if __name__ == "__main__":
    main()
