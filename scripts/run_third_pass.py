#!/usr/bin/env python3
"""Reproduce the third-pass research experiments from existing raw data."""

from __future__ import annotations

import argparse

from solana_sniper_reverse_engineering.curve_replay import run_curve_replay
from solana_sniper_reverse_engineering.third_pass import (
    run_historical_outcome_experiment,
    run_developer_sell_outcome_experiment,
    run_profitable_disagreement_experiment,
    run_ranking_hard_negative_experiment,
    run_signed_message_metadata_experiment,
    run_weekly_regime_analysis,
)


STAGES = {
    "historical": run_historical_outcome_experiment,
    "historical_sell": run_developer_sell_outcome_experiment,
    "regime": run_weekly_regime_analysis,
    "ranking": run_ranking_hard_negative_experiment,
    "strategy": run_profitable_disagreement_experiment,
    "execution": run_curve_replay,
    "metadata": run_signed_message_metadata_experiment,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(*STAGES, "all"), nargs="?", default="all")
    parser.add_argument("--force", action="store_true", help="rebuild deterministic feature caches")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for name, function in selected.items():
        print(f"\n=== third pass: {name} ===")
        if name in {"historical", "historical_sell", "execution", "metadata"}:
            function(force=args.force)
        else:
            function()


if __name__ == "__main__":
    main()
