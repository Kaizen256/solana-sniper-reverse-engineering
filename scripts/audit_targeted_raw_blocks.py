from __future__ import annotations

import argparse
from pathlib import Path

from solana_sniper_reverse_engineering.raw_block_audit import (
    run_targeted_raw_block_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    run_targeted_raw_block_audit(args.batch)


if __name__ == "__main__":
    main()
