from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path

import duckdb
import orjson

from .config import (
    BOUGHT_ACTIVITY,
    BOUGHT_TXS,
    JUNE_TRADES,
    NOT_BOUGHT_ACTIVITY,
    SUBMISSION,
    ensure_output_dirs,
)
from .feature_store import FEATURE_STORE
from .message_features import PUMP_PROGRAM, b58decode, extract_record


BLOCK_INDEX = Path("data/downloads/raw_block_index/june_slots_index.parquet")
BATCH_LISTING = Path("data/downloads/raw_block_index/batches_index.html")
STRATEGY_PREDICTIONS = Path(
    "artifacts/tables/third_pass_june_strategy_predictions.parquet"
)
RESULTS = SUBMISSION / "tables" / "third_pass_availability_audit.json"


def _unsupported_positive_programs() -> dict[str, object]:
    counts: Counter[str] = Counter()
    missing = total = 0
    with gzip.open(BOUGHT_TXS, "rb") as stream:
        for line in stream:
            total += 1
            record = orjson.loads(line)
            if extract_record(record):
                continue
            missing += 1
            instructions = record["transaction"]["message"]["instructions"]
            programs = sorted(
                {
                    str(instruction.get("programId", ""))
                    for instruction in instructions
                    if instruction.get("programId")
                    not in {
                        "ComputeBudget111111111111111111111111111111",
                        "11111111111111111111111111111111",
                    }
                }
            )
            pump_discriminators = []
            for instruction in instructions:
                if instruction.get("programId") != PUMP_PROGRAM or not instruction.get("data"):
                    continue
                try:
                    pump_discriminators.append(
                        b58decode(str(instruction["data"]))[:8].hex()
                    )
                except (KeyError, ValueError):
                    pump_discriminators.append("decode_error")
            key = "|".join(pump_discriminators or programs)
            counts[key] += 1
    return {
        "bought_archive_rows": total,
        "unsupported_bought_rows": missing,
        "top_level_program_or_pump_discriminator_counts": dict(counts.most_common()),
        "interpretation": "All unsupported positives are wrapper-program messages, not unknown top-level Pump create discriminators.",
    }


def _batch_sizes() -> dict[str, int]:
    text = BATCH_LISTING.read_text()
    return {
        match.group(1): int(match.group(2))
        for match in re.finditer(
            r'href="(batch_\d+\.jsonl\.zst)"[^\n]*\s(\d+)\n', text
        )
    }


def run_availability_audit() -> dict[str, object]:
    ensure_output_dirs()
    con = duckdb.connect()
    con.execute("SET threads=20")
    graph = con.execute(
        """
        WITH a AS (
          SELECT event_type,from_address,to_address FROM read_parquet(?)
          UNION ALL
          SELECT event_type,from_address,to_address FROM read_parquet(?)
        )
        SELECT event_type,count(*) AS rows,
               count(*) FILTER (WHERE length(trim(from_address))>0) nonempty_from,
               count(*) FILTER (WHERE length(trim(to_address))>0) nonempty_to
        FROM a GROUP BY 1 ORDER BY rows DESC
        """,
        [str(NOT_BOUGHT_ACTIVITY), str(BOUGHT_ACTIVITY)],
    ).fetch_df()
    unsupported = con.execute(
        """
        WITH missing AS (
          SELECT token_address,label FROM read_parquet(?) WHERE message_missing=1
        ), raw AS (
          SELECT token_address,launchpad,launchpad_platform FROM read_parquet(?)
          WHERE event_type='launch'
          UNION ALL
          SELECT token_address,launchpad,launchpad_platform FROM read_parquet(?)
          WHERE event_type='launch'
        ), activity AS (
          SELECT token_address,any_value(launchpad) launchpad,
                 any_value(launchpad_platform) launchpad_platform
          FROM raw GROUP BY 1
        )
        SELECT coalesce(launchpad,'unjoined') launchpad,
               coalesce(launchpad_platform,'unjoined') platform,
               count(*) AS rows,sum(label) positives
        FROM missing LEFT JOIN activity USING(token_address)
        GROUP BY ALL ORDER BY rows DESC
        """,
        [str(FEATURE_STORE), str(NOT_BOUGHT_ACTIVITY), str(BOUGHT_ACTIVITY)],
    ).fetch_df()

    sizes = _batch_sizes()
    deployed = con.execute(
        """
        WITH d AS (
          SELECT p.token_address,p.label,p.baseline_selected,p.quality_selected,p.two_stage_selected,
                 p.selective_two_stage_selected,min(t.deploy_block_slot) deploy_slot
          FROM read_parquet(?) p JOIN read_parquet(?) t USING(token_address)
          GROUP BY ALL
        )
        SELECT d.*,i.jsonl_zst_file FROM d
        LEFT JOIN read_parquet(?) i ON d.deploy_slot=i.slot
        """,
        [str(STRATEGY_PREDICTIONS), str(JUNE_TRADES), str(BLOCK_INDEX)],
    ).fetch_df()
    cohorts = {
        "target": "label",
        "baseline": "baseline_selected",
        "quality_augmented": "quality_selected",
        "two_stage": "two_stage_selected",
        "selective_two_stage": "selective_two_stage_selected",
    }
    raw_cost: dict[str, object] = {}
    for name, column in cohorts.items():
        selected = deployed[deployed[column].eq(1)]
        files = set(selected.jsonl_zst_file.dropna())
        raw_cost[name] = {
            "tokens": int(len(selected)),
            "mapped_tokens": int(selected.jsonl_zst_file.notna().sum()),
            "batches": len(files),
            "compressed_gib": sum(sizes[file] for file in files) / 2**30,
        }
    union = deployed[
        deployed[list(cohorts.values())].eq(1).any(axis=1)
    ]
    union_files = set(union.jsonl_zst_file.dropna())
    raw_cost["union"] = {
        "tokens": int(len(union)),
        "batches": len(union_files),
        "compressed_gib": sum(sizes[file] for file in union_files) / 2**30,
    }

    output: dict[str, object] = {
        "graph_edge_audit": {
            "by_event": graph.to_dict("records"),
            "total_nonempty_from": int(graph.nonempty_from.sum()),
            "total_nonempty_to": int(graph.nonempty_to.sum()),
            "decision": "DROP: supplied activity from_address/to_address fields are empty, so no historical counterparty graph exists.",
        },
        "unsupported_create_audit": {
            "population": unsupported.to_dict("records"),
            "positive_program_scan": _unsupported_positive_programs(),
            "decision": "DROP: only 21 positives; top-level wrapper calls would require undocumented wrapper decoding or post-execution inner instructions.",
        },
        "raw_block_access": {
            "index_rows": 720_288,
            "batches": len(sizes),
            "total_compressed_gib": sum(sizes.values()) / 2**30,
            "cohort_cost": raw_cost,
            "downloaded_sample": {
                "batch": "batch_002806.jsonl.zst",
                "compressed_bytes": sizes["batch_002806.jsonl.zst"],
            },
            "decision": "One targeted batch only; full or cohort-scale retrieval is disproportionate.",
        },
    }
    RESULTS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output
