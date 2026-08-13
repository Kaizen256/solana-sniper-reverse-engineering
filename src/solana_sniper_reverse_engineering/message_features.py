from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .config import BOUGHT_TXS, INTERIM, NOT_BOUGHT_TXS, ensure_output_dirs


PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
COMPUTE_BUDGET_PROGRAM = "ComputeBudget111111111111111111111111111111"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
CREATE_DISCRIMINATOR = bytes.fromhex("d6904cec5f8b31b4")
CREATE_V2_DISCRIMINATOR = bytes.fromhex("181ec828051c0777")
BUY_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")
BUY_EXACT_SOL_DISCRIMINATOR = bytes.fromhex("38fc74089edfcd5f")
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_VALUES = {char: index for index, char in enumerate(_B58_ALPHABET)}
_SPACE = re.compile(r"\s+")


SCHEMA = pa.schema(
    [
        ("tx_hash", pa.string()),
        ("token_address", pa.string()),
        ("block_time", pa.int64()),
        ("block_slot", pa.int64()),
        ("transaction_index", pa.int32()),
        ("version", pa.string()),
        ("name", pa.string()),
        ("symbol", pa.string()),
        ("uri", pa.string()),
        ("name_normalized", pa.string()),
        ("symbol_normalized", pa.string()),
        ("uri_host", pa.string()),
        ("uri_provider", pa.string()),
        ("name_length", pa.int16()),
        ("symbol_length", pa.int16()),
        ("uri_length", pa.int16()),
        ("name_word_count", pa.int16()),
        ("name_digit_count", pa.int16()),
        ("symbol_digit_count", pa.int16()),
        ("name_non_ascii_count", pa.int16()),
        ("symbol_non_ascii_count", pa.int16()),
        ("name_upper_fraction", pa.float32()),
        ("symbol_upper_fraction", pa.float32()),
        ("name_symbol_same_normalized", pa.int8()),
        ("name_has_url", pa.int8()),
        ("name_has_dollar", pa.int8()),
        ("name_has_emoji_or_non_ascii", pa.int8()),
        ("symbol_has_emoji_or_non_ascii", pa.int8()),
        ("n_message_instructions", pa.int16()),
        ("n_account_keys", pa.int16()),
        ("n_signers", pa.int16()),
        ("n_writable_accounts", pa.int16()),
        ("n_address_table_lookups", pa.int16()),
        ("n_compute_budget_instructions", pa.int16()),
        ("compute_unit_limit", pa.int64()),
        ("compute_unit_price_micro_lamports", pa.uint64()),
        ("n_system_transfers", pa.int16()),
        ("system_transfer_lamports", pa.uint64()),
        ("max_system_transfer_lamports", pa.uint64()),
        ("n_pump_instructions", pa.int16()),
        ("n_create_instructions", pa.int16()),
        ("create_instruction_index", pa.int16()),
        ("has_dev_buy", pa.int8()),
        ("dev_buy_kind", pa.string()),
        ("dev_buy_lamports", pa.uint64()),
        ("dev_buy_token_amount_raw", pa.uint64()),
        ("dev_buy_min_tokens_raw", pa.uint64()),
        ("dev_buy_instruction_index", pa.int16()),
    ]
)


def b58decode(value: str) -> bytes:
    number = 0
    for char in value:
        number = number * 58 + _B58_VALUES[char]
    raw = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    return b"\0" * (len(value) - len(value.lstrip("1"))) + raw


def _borsh_string(payload: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(payload):
        raise ValueError("missing Borsh string length")
    length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    if length > 10_000 or offset + length > len(payload):
        raise ValueError("invalid Borsh string length")
    return payload[offset : offset + length].decode("utf-8", errors="replace"), offset + length


def parse_create(data: str) -> tuple[str, str, str] | None:
    payload = b58decode(data)
    if payload[:8] not in (CREATE_DISCRIMINATOR, CREATE_V2_DISCRIMINATOR):
        return None
    try:
        name, offset = _borsh_string(payload, 8)
        symbol, offset = _borsh_string(payload, offset)
        uri, _ = _borsh_string(payload, offset)
    except ValueError:
        return None
    return name, symbol, uri


def _provider(uri: str) -> tuple[str, str]:
    lowered = uri.lower().strip()
    try:
        host = (urlparse(lowered).hostname or "").removeprefix("www.")
    except ValueError:
        host = ""
    if "ipfs" in lowered:
        provider = "ipfs"
    elif "arweave" in lowered or "arweave.net" in host:
        provider = "arweave"
    elif "pinata" in lowered:
        provider = "pinata"
    elif "uxento" in lowered:
        provider = "uxento"
    elif "pump" in host:
        provider = "pump"
    elif host:
        provider = "other_http"
    else:
        provider = "missing_or_non_http"
    return host, provider


def _text_features(name: str, symbol: str, uri: str) -> dict[str, object]:
    normalized_name = _SPACE.sub(" ", name.strip().lower())
    normalized_symbol = _SPACE.sub("", symbol.strip().lower())
    host, provider = _provider(uri)

    def upper_fraction(value: str) -> float:
        letters = [char for char in value if char.isalpha()]
        return float(sum(char.isupper() for char in letters) / len(letters)) if letters else 0.0

    return {
        "name": name,
        "symbol": symbol,
        "uri": uri,
        "name_normalized": normalized_name,
        "symbol_normalized": normalized_symbol,
        "uri_host": host,
        "uri_provider": provider,
        "name_length": min(len(name), 32_767),
        "symbol_length": min(len(symbol), 32_767),
        "uri_length": min(len(uri), 32_767),
        "name_word_count": min(len(name.split()), 32_767),
        "name_digit_count": min(sum(char.isdigit() for char in name), 32_767),
        "symbol_digit_count": min(sum(char.isdigit() for char in symbol), 32_767),
        "name_non_ascii_count": min(sum(ord(char) > 127 for char in name), 32_767),
        "symbol_non_ascii_count": min(sum(ord(char) > 127 for char in symbol), 32_767),
        "name_upper_fraction": upper_fraction(name),
        "symbol_upper_fraction": upper_fraction(symbol),
        "name_symbol_same_normalized": int(normalized_name.replace(" ", "") == normalized_symbol),
        "name_has_url": int("http://" in normalized_name or "https://" in normalized_name or ".com" in normalized_name),
        "name_has_dollar": int("$" in name or "$" in symbol),
        "name_has_emoji_or_non_ascii": int(any(ord(char) > 127 for char in name)),
        "symbol_has_emoji_or_non_ascii": int(any(ord(char) > 127 for char in symbol)),
    }


def _u64(payload: bytes, offset: int) -> int | None:
    return struct.unpack_from("<Q", payload, offset)[0] if len(payload) >= offset + 8 else None


def extract_record(record: dict[str, object]) -> list[dict[str, object]]:
    transaction = record["transaction"]  # type: ignore[index]
    message = transaction["message"]  # type: ignore[index]
    instructions = message.get("instructions", [])
    keys = message.get("accountKeys", [])
    transaction_features: dict[str, object] = {
        "tx_hash": transaction["signatures"][0],
        "block_time": record["blockTime"],
        "block_slot": record["slot"],
        "transaction_index": record["transactionIndex"],
        "version": str(record.get("version", "")),
        "n_message_instructions": len(instructions),
        "n_account_keys": len(keys),
        "n_signers": sum(bool(key.get("signer")) for key in keys),
        "n_writable_accounts": sum(bool(key.get("writable")) for key in keys),
        "n_address_table_lookups": len(message.get("addressTableLookups") or []),
        "n_compute_budget_instructions": 0,
        "compute_unit_limit": None,
        "compute_unit_price_micro_lamports": None,
        "n_system_transfers": 0,
        "system_transfer_lamports": 0,
        "max_system_transfer_lamports": 0,
        "n_pump_instructions": 0,
    }
    creates: list[dict[str, object]] = []
    buys: list[dict[str, object]] = []
    for instruction_index, instruction in enumerate(instructions):
        program_id = instruction.get("programId", "")
        if program_id == COMPUTE_BUDGET_PROGRAM:
            transaction_features["n_compute_budget_instructions"] += 1  # type: ignore[operator]
            try:
                payload = b58decode(instruction.get("data", ""))
                if payload and payload[0] == 2 and len(payload) >= 5:
                    transaction_features["compute_unit_limit"] = struct.unpack_from("<I", payload, 1)[0]
                elif payload and payload[0] == 3 and len(payload) >= 9:
                    transaction_features["compute_unit_price_micro_lamports"] = struct.unpack_from("<Q", payload, 1)[0]
            except (KeyError, ValueError):
                pass
        if program_id == SYSTEM_PROGRAM:
            parsed = instruction.get("parsed") or {}
            if parsed.get("type") == "transfer":
                lamports = int((parsed.get("info") or {}).get("lamports") or 0)
                transaction_features["n_system_transfers"] += 1  # type: ignore[operator]
                transaction_features["system_transfer_lamports"] += lamports  # type: ignore[operator]
                transaction_features["max_system_transfer_lamports"] = max(
                    int(transaction_features["max_system_transfer_lamports"]), lamports
                )
        if program_id != PUMP_PROGRAM:
            continue
        transaction_features["n_pump_instructions"] += 1  # type: ignore[operator]
        data = instruction.get("data")
        accounts = instruction.get("accounts") or []
        if not data:
            continue
        try:
            payload = b58decode(data)
        except (KeyError, ValueError):
            continue
        create = parse_create(data)
        if create is not None and accounts:
            name, symbol, uri = create
            creates.append(
                {
                    "token_address": accounts[0],
                    "create_instruction_index": instruction_index,
                    **_text_features(name, symbol, uri),
                }
            )
        elif payload.startswith(BUY_DISCRIMINATOR) and len(accounts) > 2:
            buys.append(
                {
                    "token_address": accounts[2],
                    "dev_buy_kind": "buy_max_sol_cost",
                    "dev_buy_lamports": _u64(payload, 16),
                    "dev_buy_token_amount_raw": _u64(payload, 8),
                    "dev_buy_min_tokens_raw": None,
                    "dev_buy_instruction_index": instruction_index,
                }
            )
        elif payload.startswith(BUY_EXACT_SOL_DISCRIMINATOR) and len(accounts) > 2:
            buys.append(
                {
                    "token_address": accounts[2],
                    "dev_buy_kind": "buy_exact_sol_in",
                    "dev_buy_lamports": _u64(payload, 8),
                    "dev_buy_token_amount_raw": None,
                    "dev_buy_min_tokens_raw": _u64(payload, 16),
                    "dev_buy_instruction_index": instruction_index,
                }
            )
    buys_by_token = {str(buy["token_address"]): buy for buy in buys}
    output: list[dict[str, object]] = []
    for create in creates:
        token = str(create["token_address"])
        buy = buys_by_token.get(token)
        output.append(
            {
                **transaction_features,
                **create,
                "n_create_instructions": len(creates),
                "has_dev_buy": int(buy is not None),
                "dev_buy_kind": buy["dev_buy_kind"] if buy else "none",
                "dev_buy_lamports": buy["dev_buy_lamports"] if buy else 0,
                "dev_buy_token_amount_raw": buy["dev_buy_token_amount_raw"] if buy else None,
                "dev_buy_min_tokens_raw": buy["dev_buy_min_tokens_raw"] if buy else None,
                "dev_buy_instruction_index": buy["dev_buy_instruction_index"] if buy else None,
            }
        )
    return output


def extract_file(source: Path, destination: Path, force: bool = False) -> dict[str, object]:
    manifest_path = destination.with_suffix(".manifest.json")
    if destination.exists() and manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd", use_dictionary=True)
    batch: list[dict[str, object]] = []
    lines = rows = missing_create_lines = 0
    started = time.monotonic()
    try:
        with gzip.open(source, "rb") as stream:
            for line in stream:
                lines += 1
                extracted = extract_record(orjson.loads(line))
                if not extracted:
                    missing_create_lines += 1
                batch.extend(extracted)
                rows += len(extracted)
                if len(batch) >= 20_000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    batch.clear()
                if lines % 250_000 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"{source.name}: {lines:,} lines, {rows:,} token rows, "
                        f"{lines / elapsed:,.0f} lines/s",
                        file=sys.stderr,
                        flush=True,
                    )
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    finally:
        writer.close()
    os.replace(temporary, destination)
    manifest = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "destination": str(destination),
        "line_count": lines,
        "token_row_count": rows,
        "lines_without_supported_create": missing_create_lines,
        "strict_role": "candidate signed-message features; no transaction meta fields",
        "command": "python -m solana_sniper_reverse_engineering.message_features",
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run(force: bool = False, only: str = "all") -> list[dict[str, object]]:
    ensure_output_dirs()
    out_dir = INTERIM / "message_features"
    jobs = []
    if only in ("all", "bought"):
        jobs.append((BOUGHT_TXS, out_dir / "bought.parquet"))
    if only in ("all", "not_bought"):
        jobs.append((NOT_BOUGHT_TXS, out_dir / "not_bought.parquet"))
    manifests = [extract_file(source, destination, force=force) for source, destination in jobs]
    print(json.dumps(manifests, indent=2, sort_keys=True))
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract legal signed deployment-message features")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", choices=("all", "bought", "not_bought"), default="all")
    args = parser.parse_args()
    run(force=args.force, only=args.only)


if __name__ == "__main__":
    main()
