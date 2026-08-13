from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Solana sniper competition pipeline")
    parser.add_argument(
        "stage",
        choices=("behavior", "messages", "features", "model", "backtest", "robustness", "audit", "all"),
        nargs="?",
        default="all",
    )
    args = parser.parse_args()

    if args.stage in ("behavior", "all"):
        from .behavior import run

        run()
    if args.stage in ("messages", "all"):
        from .message_features import run

        run()
    if args.stage in ("features", "all"):
        from .feature_store import run

        feature_path = run()
    else:
        from .feature_store import FEATURE_STORE

        feature_path = FEATURE_STORE
    if args.stage in ("model", "all"):
        from .modeling import run

        run(feature_path)
    if args.stage in ("backtest", "all"):
        from .backtest import run

        run(force=True)
    if args.stage in ("robustness", "all"):
        from .robustness import run

        run()
    if args.stage in ("audit", "all"):
        from .methodological_audit import run

        run(feature_path)
