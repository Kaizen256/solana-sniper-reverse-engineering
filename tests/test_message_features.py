from __future__ import annotations

import struct

from solana_sniper_reverse_engineering.message_features import (
    CREATE_DISCRIMINATOR,
    PUMP_PROGRAM,
    extract_record,
    parse_create,
)


ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58encode(payload: bytes) -> str:
    number = int.from_bytes(payload, "big")
    encoded = ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = ALPHABET[remainder] + encoded
    return "1" * (len(payload) - len(payload.lstrip(b"\0"))) + (encoded or "")


def create_data(name: str, symbol: str, uri: str) -> str:
    payload = bytearray(CREATE_DISCRIMINATOR)
    for value in (name, symbol, uri):
        encoded = value.encode()
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)
    payload.extend(bytes(32))
    return b58encode(bytes(payload))


def test_create_metadata_round_trip() -> None:
    data = create_data("A Token", "ATK", "https://example.com/meta")
    assert parse_create(data) == ("A Token", "ATK", "https://example.com/meta")


def test_multi_token_transaction_preserves_composite_grain() -> None:
    record = {
        "blockTime": 100,
        "slot": 200,
        "transactionIndex": 3,
        "version": 0,
        "transaction": {
            "signatures": ["same-tx"],
            "message": {
                "accountKeys": [{"signer": True, "writable": True}],
                "addressTableLookups": [],
                "instructions": [
                    {
                        "programId": PUMP_PROGRAM,
                        "accounts": ["token-a"],
                        "data": create_data("A", "A", "https://a.example"),
                    },
                    {
                        "programId": PUMP_PROGRAM,
                        "accounts": ["token-b"],
                        "data": create_data("B", "B", "https://b.example"),
                    },
                ],
            },
        },
    }
    rows = extract_record(record)
    assert {(row["tx_hash"], row["token_address"]) for row in rows} == {
        ("same-tx", "token-a"),
        ("same-tx", "token-b"),
    }
    assert all(row["n_create_instructions"] == 2 for row in rows)
