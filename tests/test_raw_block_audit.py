from __future__ import annotations

import base64
import struct

from solana_sniper_reverse_engineering.raw_block_audit import (
    TRADE_EVENT_DISCRIMINATOR,
    decode_trade_event,
)


def _pubkey(fill: int) -> bytes:
    return bytes([fill]) * 32


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<I", len(encoded)) + encoded


def test_trade_event_borsh_decoder_recovers_execution_fields() -> None:
    payload = bytearray(TRADE_EVENT_DISCRIMINATOR)
    payload.extend(_pubkey(1))
    payload.extend(struct.pack("<QQ?", 1_975_308_641, 60_000_000_000_000, True))
    payload.extend(_pubkey(2))
    payload.extend(struct.pack("<qQQQQ", 123, 31_975_308_641, 1_013_000_000_000_000, 1_975_308_641, 733_100_000_000_000))
    payload.extend(_pubkey(3))
    payload.extend(struct.pack("<QQ", 95, 18_765_432))
    payload.extend(_pubkey(4))
    payload.extend(struct.pack("<QQ?QQQq", 30, 5_925_926, True, 7, 8, 9, 122))
    payload.extend(_string("buy_exact_sol_in"))
    payload.extend(struct.pack("<?", False))
    event = decode_trade_event(base64.b64encode(payload).decode())
    assert event is not None
    assert event["sol_amount_lamports"] == 1_975_308_641
    assert event["token_amount_raw"] == 60_000_000_000_000
    assert event["virtual_sol_reserves_lamports"] == 31_975_308_641
    assert event["protocol_fee_bps"] == 95
    assert event["creator_fee_bps"] == 30
    assert event["ix_name"] == "buy_exact_sol_in"
    assert event["mayhem_mode"] is False


def test_trade_event_decoder_rejects_other_events() -> None:
    encoded = base64.b64encode(bytes(64)).decode()
    assert decode_trade_event(encoded) is None
