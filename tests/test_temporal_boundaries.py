from __future__ import annotations

import duckdb


def test_activity_asof_is_strictly_before_deployment() -> None:
    con = duckdb.connect()
    result = con.execute(
        """
        WITH deployments(wallet, decision_time) AS (VALUES ('w', 100)),
        state(wallet, timestamp, cumulative_events) AS (
          VALUES ('w', 90, 1), ('w', 99, 2), ('w', 100, 3), ('w', 101, 4)
        )
        SELECT s.timestamp, s.cumulative_events
        FROM deployments d
        ASOF LEFT JOIN state s
          ON d.wallet=s.wallet AND d.decision_time>s.timestamp
        """
    ).fetchone()
    assert result == (99, 2)


def test_same_second_deployments_do_not_count_each_other() -> None:
    con = duckdb.connect()
    rows = con.execute(
        """
        WITH deployments(wallet, decision_time) AS (
          VALUES ('w', 90), ('w', 100), ('w', 100), ('w', 101)
        ), grouped AS (
          SELECT wallet, decision_time, count(*) n
          FROM deployments GROUP BY 1,2
        )
        SELECT decision_time,
          coalesce(sum(n) OVER (
            PARTITION BY wallet ORDER BY decision_time
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) prior_count
        FROM grouped ORDER BY decision_time
        """
    ).fetchall()
    assert rows == [(90, 0), (100, 1), (101, 3)]


def test_rolling_window_includes_boundary_but_not_decision_second() -> None:
    con = duckdb.connect()
    value = con.execute(
        """
        WITH events(t, n) AS (VALUES (0, 1), (1, 1), (86400, 1))
        SELECT rolling FROM (
          SELECT t, sum(n) OVER (
            ORDER BY t RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
          ) rolling
          FROM events
        ) WHERE t=86400
        """
    ).fetchone()[0]
    assert value == 2


def test_outcome_horizon_is_not_observable_until_maturity() -> None:
    con = duckdb.connect()
    rows = con.execute(
        """
        WITH launches(token,launch_time,claim_time) AS (
          VALUES ('early',0,86399),('late',0,86401)
        ), maturity AS (
          SELECT token,86400 AS observable_time,
                 (claim_time<=86400)::INT AS success_within_1d
          FROM launches
        ), decisions(t) AS (VALUES (86399),(86400),(86401))
        SELECT d.t,coalesce(sum(m.success_within_1d),0) successes
        FROM decisions d LEFT JOIN maturity m ON d.t>m.observable_time
        GROUP BY d.t ORDER BY d.t
        """
    ).fetchall()
    assert rows == [(86399, 0), (86400, 0), (86401, 1)]
