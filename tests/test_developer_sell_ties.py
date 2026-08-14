from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from solana_sniper_reverse_engineering.third_pass import (
    DAY,
    _latest_dev_sell_group_feature_expressions,
    _latest_dev_sell_group_sql,
)


def _summaries(lifecycle: pd.DataFrame) -> pd.DataFrame:
    lifecycle = lifecycle.copy()
    candidates = pd.DataFrame(
        {
            "candidate": ["before_second_sell", "after_maturity"],
            "wallet": ["wallet", "wallet"],
            "block_time": [130, 100 + DAY + 1],
        }
    )
    con = duckdb.connect()
    con.register("lifecycle", lifecycle)
    con.register("candidates", candidates)
    grouped = _latest_dev_sell_group_sql("lifecycle")
    expressions = _latest_dev_sell_group_feature_expressions()
    selected = ",".join(f"{expression} AS {name}" for name, expression in expressions.items())
    return con.execute(
        f"""
        WITH latest_groups AS ({grouped})
        SELECT f.candidate,l.latest_launch_group_size,{selected}
        FROM candidates f
        ASOF LEFT JOIN latest_groups l
          ON f.wallet=l.wallet AND f.block_time>l.launch_time
        ORDER BY f.candidate
        """
    ).fetch_df()


def test_equal_timestamp_latest_launches_are_order_invariant() -> None:
    lifecycle = pd.DataFrame(
        {
            "wallet": ["wallet", "wallet", "wallet"],
            "token_address": ["first", "second", "third"],
            "launch_time": [100, 100, 100],
            "first_dev_sell_time": [110.0, 140.0, np.nan],
        }
    )
    forward = _summaries(lifecycle)
    reversed_rows = _summaries(lifecycle.iloc[::-1].reset_index(drop=True).copy())
    pd.testing.assert_frame_equal(forward, reversed_rows)


def test_equal_timestamp_latest_launches_use_symmetric_group_semantics() -> None:
    lifecycle = pd.DataFrame(
        {
            "wallet": ["wallet", "wallet", "wallet"],
            "token_address": ["first", "second", "third"],
            "launch_time": [100, 100, 100],
            "first_dev_sell_time": [110.0, 140.0, np.nan],
        }
    )
    result = _summaries(lifecycle).set_index("candidate")
    before = result.loc["before_second_sell"]
    assert before.latest_launch_group_size == 3
    assert before.latest_prior_launch_group_dev_sold_fraction == 1 / 3
    assert before.latest_prior_launch_group_mature_1d_no_dev_sell_fraction == 0
    assert before.latest_prior_launch_group_mature_7d_no_dev_sell_fraction == 0
    assert before.latest_prior_launch_group_dev_sell_latency_median_seconds == 10

    mature = result.loc["after_maturity"]
    assert mature.latest_prior_launch_group_dev_sold_fraction == 2 / 3
    assert mature.latest_prior_launch_group_mature_1d_no_dev_sell_fraction == 1 / 3
    assert mature.latest_prior_launch_group_mature_7d_no_dev_sell_fraction == 0
    assert mature.latest_prior_launch_group_dev_sell_latency_median_seconds == 25
