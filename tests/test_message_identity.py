from __future__ import annotations

import json
from pathlib import Path

import pytest

from solana_sniper_reverse_engineering.message_identity import _message_fingerprints


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_REPORT = ROOT / "submission/tables/message_identity_validation.json"


def test_message_fingerprints_use_signed_message_fields_and_exclude_current_token() -> None:
    token = "Mint111"
    message = {
        "accountKeys": [
            {"pubkey": "FeePayer", "signer": True, "writable": True, "source": "transaction"},
            {"pubkey": token, "signer": True, "writable": True, "source": "transaction"},
            {"pubkey": "CoSigner", "signer": True, "writable": False, "source": "transaction"},
            {"pubkey": "LookupWritable", "signer": False, "writable": True, "source": "lookupTable"},
        ],
        "addressTableLookups": [{"accountKey": "AltTable"}],
        "instructions": [
            {"programId": "ProgramA"},
            {
                "programId": "11111111111111111111111111111111",
                "parsed": {"type": "transfer", "info": {"destination": "TipDestination"}},
            },
        ],
    }
    fingerprints = set(_message_fingerprints(message, [token]))
    assert ("alt_table", "AltTable") in fingerprints
    assert ("system_transfer_destination", "TipDestination") in fingerprints
    assert ("extra_signer", "CoSigner") in fingerprints
    assert ("lookup_writable", "LookupWritable") in fingerprints
    assert ("extra_signer", token) not in fingerprints
    assert ("extra_signer", "FeePayer") not in fingerprints
    assert any(kind == "program_set" for kind, _ in fingerprints)
    assert any(kind == "program_sequence" for kind, _ in fingerprints)


@pytest.mark.skipif(not VALIDATION_REPORT.exists(), reason="message identity validation not present")
def test_message_identity_family_was_rejected_before_june() -> None:
    result = json.loads(VALIDATION_REPORT.read_text())
    assert result["status"] == "REJECT"
    assert result["june_opened"] is False
    assert result["selected_model"] is None
    assert result["windows"]["may"]["baseline_plus_all_message_identities_pr_auc_delta"] < 0
