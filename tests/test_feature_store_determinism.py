from __future__ import annotations

import duckdb
import pandas as pd

from solana_sniper_reverse_engineering.feature_store import ACTIVITY_AMOUNT_DECIMAL
from solana_sniper_reverse_engineering.modeling import _load_split


def test_activity_amount_accumulation_is_order_invariant() -> None:
    """String-valued monetary inputs must not inherit binary-float sum order."""
    values = ["56.997300987826", "0.00000000000001", "-0.00000000000001"]
    connection = duckdb.connect()
    observed = []
    for ordered in (values, list(reversed(values))):
        placeholders = ", ".join(f"('{value}')" for value in ordered)
        result = connection.execute(
            f"""
            SELECT cast(sum(try_cast(value AS {ACTIVITY_AMOUNT_DECIMAL})) AS DOUBLE)
            FROM (VALUES {placeholders}) AS source(value)
            """
        ).fetchone()[0]
        observed.append(result)

    assert ACTIVITY_AMOUNT_DECIMAL == "DECIMAL(18,9)"
    assert observed == [56.997300988, 56.997300988]


def test_model_loader_is_invariant_to_parquet_physical_row_order(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "token_address": ["token-c", "token-a", "token-b"],
            "tx_hash": ["tx-c", "tx-a", "tx-b"],
            "block_time": [1_773_273_602, 1_773_273_600, 1_773_273_601],
            "label": [1, 1, 1],
        }
    )
    forward = tmp_path / "forward.parquet"
    reverse = tmp_path / "reverse.parquet"
    rows.to_parquet(forward, index=False)
    rows.iloc[::-1].to_parquet(reverse, index=False)

    loaded_forward = _load_split(forward, [], [], "train", 2)
    loaded_reverse = _load_split(reverse, [], [], "train", 2)
    assert loaded_forward.equals(loaded_reverse)
    assert loaded_forward.token_address.tolist() == ["token-a", "token-b", "token-c"]
