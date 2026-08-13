#!/usr/bin/env python3
"""Run the bounded integer Pump bonding-curve replay."""

from __future__ import annotations

import argparse

from solana_sniper_reverse_engineering.curve_replay import run_curve_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_curve_replay(force=args.force)


if __name__ == "__main__":
    main()
