from __future__ import annotations

import base64
import json
import struct
import subprocess
from pathlib import Path
from typing import BinaryIO

import duckdb
import pandas as pd

from .config import ARTIFACTS, JUNE_TRADES, SUBMISSION, ensure_output_dirs
SLOT_INDEX = Path("data/downloads/raw_block_index/june_slots_index.parquet")
STRATEGY_PREDICTIONS = ARTIFACTS / "tables" / "third_pass_june_strategy_predictions.parquet"
RAW_AUDIT_RESULTS = SUBMISSION / "tables" / "targeted_raw_block_audit.json"
RAW_EVENT_TABLE = ARTIFACTS / "tables" / "targeted_raw_block_trade_events.csv"
TRADE_EVENT_DISCRIMINATOR = bytes([189, 219, 127, 211, 78, 230, 97, 238])
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58_encode(value: bytes) -> str:
    zeros = len(value) - len(value.lstrip(b"\0"))
    number = int.from_bytes(value, "big")
    encoded = ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = BASE58_ALPHABET[remainder] + encoded
    return "1" * zeros + encoded


class _BorshReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("truncated event")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def boolean(self) -> bool:
        return bool(self.take(1)[0])

    def pubkey(self) -> str:
        return _base58_encode(self.take(32))

    def string(self) -> str:
        size = struct.unpack("<I", self.take(4))[0]
        return self.take(size).decode("utf-8")


def decode_trade_event(encoded: str) -> dict[str, object] | None:
    raw = base64.b64decode(encoded)
    if not raw.startswith(TRADE_EVENT_DISCRIMINATOR):
        return None
    reader = _BorshReader(raw[len(TRADE_EVENT_DISCRIMINATOR) :])
    event: dict[str, object] = {
        "mint": reader.pubkey(),
        "sol_amount_lamports": reader.u64(),
        "token_amount_raw": reader.u64(),
        "is_buy": reader.boolean(),
        "user": reader.pubkey(),
        "timestamp": reader.i64(),
        "virtual_sol_reserves_lamports": reader.u64(),
        "virtual_token_reserves_raw": reader.u64(),
        "real_sol_reserves_lamports": reader.u64(),
        "real_token_reserves_raw": reader.u64(),
        "fee_recipient": reader.pubkey(),
        "protocol_fee_bps": reader.u64(),
        "protocol_fee_lamports": reader.u64(),
        "creator": reader.pubkey(),
        "creator_fee_bps": reader.u64(),
        "creator_fee_lamports": reader.u64(),
        "track_volume": reader.boolean(),
        "total_unclaimed_tokens": reader.u64(),
        "total_claimed_tokens": reader.u64(),
        "current_sol_volume": reader.u64(),
        "last_update_timestamp": reader.i64(),
        "ix_name": reader.string(),
    }
    # This field is present in the June-era event. Later trailing extensions are
    # intentionally not decoded because they are not needed for this audit.
    event["mayhem_mode"] = reader.boolean() if reader.offset < len(reader.data) else None
    return event


def _candidate_tokens(batch_name: str) -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(
        """
        WITH deployed AS (
          SELECT p.token_address,p.label,p.baseline_selected,p.quality_selected,p.two_stage_selected,
                 p.selective_two_stage_selected,min(t.deploy_block_slot) deploy_slot,
                 min(t.deploy_tx_hash) deploy_tx_hash
          FROM read_parquet(?) p JOIN read_parquet(?) t USING(token_address)
          GROUP BY ALL
        )
        SELECT d.* FROM deployed d JOIN read_parquet(?) i ON d.deploy_slot=i.slot
        WHERE i.jsonl_zst_file=? AND (
          d.label=1 OR d.baseline_selected=1 OR d.quality_selected=1 OR d.two_stage_selected=1
          OR d.selective_two_stage_selected=1
        )
        """,
        [str(STRATEGY_PREDICTIONS), str(JUNE_TRADES), str(SLOT_INDEX), batch_name],
    ).fetch_df()


def _json_lines(path: Path) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    process = subprocess.Popen(  # noqa: S603
        ["zstdcat", str(path)],  # noqa: S607
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("zstdcat stdout was unavailable")
    return process, process.stdout


def run_targeted_raw_block_audit(batch_path: Path) -> dict[str, object]:
    """Decode Pump TradeEvents for selected tokens in one targeted raw batch."""
    ensure_output_dirs()
    if not batch_path.exists():
        raise FileNotFoundError(batch_path)
    candidates = _candidate_tokens(batch_path.name)
    slot_tokens = {
        int(slot): set(frame.token_address)
        for slot, frame in candidates.groupby("deploy_slot")
    }
    candidate_set = set(candidates.token_address)
    rows: list[dict[str, object]] = []
    matched_transactions = 0
    process, stream = _json_lines(batch_path)
    try:
        for raw_line in stream:
            block = json.loads(raw_line)
            slot = int(block["slot"])
            if slot not in slot_tokens:
                continue
            for transaction in block["block"]["transactions"]:
                message = transaction["transaction"]["message"]
                keys = {item["pubkey"] for item in message["accountKeys"]}
                relevant = candidate_set.intersection(keys)
                if not relevant:
                    continue
                logs = transaction["meta"].get("logMessages") or []
                if not any(PUMP_PROGRAM in log for log in logs):
                    continue
                matched_transactions += 1
                signature = transaction["transaction"]["signatures"][0]
                instruction_names = [
                    log.removeprefix("Program log: Instruction: ")
                    for log in logs
                    if log.startswith("Program log: Instruction: ")
                ]
                for event_log_index, log in enumerate(logs):
                    if not log.startswith("Program data: "):
                        continue
                    try:
                        event = decode_trade_event(log.removeprefix("Program data: "))
                    except (ValueError, UnicodeDecodeError, struct.error):
                        continue
                    if event is None or event["mint"] not in relevant:
                        continue
                    rows.append(
                        {
                            "slot": slot,
                            "signature": signature,
                            "event_log_index": event_log_index,
                            "token_address": event.pop("mint"),
                            "instruction_logs": "|".join(instruction_names),
                            **event,
                        }
                    )
    finally:
        stream.close()
        stderr = process.stderr.read().decode("utf-8") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstdcat failed ({return_code}): {stderr}")

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("No relevant Pump TradeEvents decoded")
    con = duckdb.connect()
    con.register("events", events)
    compared = con.execute(
        """
        SELECT e.*,t.side normalized_side,t.amount_sol normalized_amount_sol,
               t.base_amount normalized_base_amount,t.price_sol normalized_price_sol,
               abs(t.amount_sol-e.sol_amount_lamports/1e9) normalized_sol_error,
               abs(t.base_amount-e.token_amount_raw/1e6) normalized_token_error
        FROM events e LEFT JOIN read_parquet(?) t
          ON e.signature=t.tx_hash AND e.token_address=t.token_address
        QUALIFY row_number() OVER (
          PARTITION BY e.signature,e.event_log_index
          ORDER BY abs(t.base_amount-e.token_amount_raw/1e6),
                   abs(t.amount_sol-e.sol_amount_lamports/1e9)
        )=1
        """,
        [str(JUNE_TRADES)],
    ).fetch_df()
    compared.to_csv(RAW_EVENT_TABLE, index=False)

    summary: dict[str, object] = {
        "batch": batch_path.name,
        "compressed_bytes": batch_path.stat().st_size,
        "candidate_tokens": int(len(candidates)),
        "candidate_cohorts": {
            "target": int(candidates.label.sum()),
            "baseline": int(candidates.baseline_selected.sum()),
            "quality_augmented": int(candidates.quality_selected.sum()),
            "two_stage": int(candidates.two_stage_selected.sum()),
            "selective_two_stage": int(candidates.selective_two_stage_selected.sum()),
        },
        "matched_transactions": matched_transactions,
        "decoded_trade_events": int(len(compared)),
        "decoded_tokens": int(compared.token_address.nunique()),
        "ix_name_counts": compared.ix_name.value_counts().sort_index().to_dict(),
        "instruction_event_counts": (
            compared.groupby(["instruction_logs", "ix_name", "mayhem_mode"])
            .size()
            .rename("events")
            .reset_index()
            .to_dict("records")
        ),
        "fee_modes": (
            compared.groupby(["protocol_fee_bps", "creator_fee_bps", "mayhem_mode"])
            .size()
            .rename("events")
            .reset_index()
            .to_dict("records")
        ),
        "normalization_audit": {
            "sol_exact_events": int(compared.normalized_sol_error.fillna(-1).eq(0).sum()),
            "token_exact_events": int(compared.normalized_token_error.fillna(-1).eq(0).sum()),
            "max_sol_error": float(compared.normalized_sol_error.max()),
            "max_token_error": float(compared.normalized_token_error.max()),
        },
        "conclusion": "One batch is enough to verify June instruction/event variants, but not enough to estimate full-cohort instruction shares.",
    }
    RAW_AUDIT_RESULTS.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary
