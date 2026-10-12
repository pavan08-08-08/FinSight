import os
import re
import hmac
import hashlib
from typing import Optional
from backend.app.services.errors import MissingHMACKeyError


from backend.app.schemas import (
    make_evidence_id as canonical_make_evidence_id,
    mask_reference as canonical_mask_reference,
    fingerprint_reference as canonical_fingerprint_reference,
)


def mask_reference(raw_ref: Optional[str]) -> Optional[str]:
    """
    Masks sensitive account, loan, or policy references for safe outward-facing display.
    Canonical format: 'XXXX' or 'XXXX' + last 4 characters.
    """
    if not raw_ref or not raw_ref.strip():
        return None
    try:
        return canonical_mask_reference(raw_ref)
    except ValueError:
        return None


from typing import Optional, Union


def compute_reference_fingerprint(
    raw_ref: Optional[str],
    hmac_key: Optional[Union[str, bytes]] = None
) -> Optional[str]:
    """
    Computes an HMAC-SHA256 hex digest of a reference using PROOFLEDGER_REFERENCE_HMAC_KEY.
    Fails closed when key is missing or empty.
    """
    if not raw_ref or not raw_ref.strip():
        return None

    secret = hmac_key if hmac_key is not None else os.environ.get("PROOFLEDGER_REFERENCE_HMAC_KEY")
    if secret is None:
        raise MissingHMACKeyError("PROOFLEDGER_REFERENCE_HMAC_KEY is not set; cannot fingerprint references")
    if isinstance(secret, str) and not secret.strip():
        raise MissingHMACKeyError("PROOFLEDGER_REFERENCE_HMAC_KEY is empty; cannot fingerprint references")
    if isinstance(secret, bytes) and not secret.strip():
        raise MissingHMACKeyError("PROOFLEDGER_REFERENCE_HMAC_KEY is empty; cannot fingerprint references")

    try:
        return canonical_fingerprint_reference(raw_ref, key=secret)
    except RuntimeError as e:
        raise MissingHMACKeyError(str(e))
    except ValueError:
        return None


def make_evidence_id(document_id: str, source_location: Optional[str], occurrence_index: int = 0) -> str:
    """Canonical evidence ID generator (Member 5 contract)."""
    return canonical_make_evidence_id(document_id, source_location, occurrence_index)


def generate_evidence_id(
    document_id: str,
    evidence_type: str = "",
    source_location_str: str = "",
    occurrence_index: int = 0
) -> str:
    """
    Adapter bridging Member 1 calls to canonical make_evidence_id.
    Ensures stable, value-free generated IDs conforming to ^ev_[0-9a-f]{20}$.
    """
    loc = source_location_str or None
    return canonical_make_evidence_id(document_id, loc, occurrence_index)



def sanitize_free_text(text: Optional[str]) -> Optional[str]:
    """
    Sanitizes free-text narration by redacting likely sensitive numeric identifiers:
    - Labeled account numbers (e.g. 'A/C 123456789012', 'AC: 9988776655')
    - 10-digit mobile/phone numbers (e.g. 'Mob 9876543210')
    - Contiguous sequences of 9+ digits (card, account, transaction, or UTR IDs)
    Preserves merchant names, ordinary transaction descriptions, and words.
    Does NOT claim full PII removal (names/addresses remain).
    """
    if not text:
        return text
    
    cleaned = str(text)

    # 1. Redact standalone alphanumeric reference patterns (e.g. 'FD-30451122', 'LN-987654321', 'POL12345678', 'REF99887766')
    cleaned = re.sub(
        r'\b([A-Za-z]{2,5})[-/](\d{4,16})\b',
        lambda m: mask_reference(m.group(0)) or "XXXX",
        cleaned
    )
    cleaned = re.sub(
        r'\b([A-Za-z]{2,4})(\d{6,16})\b',
        lambda m: mask_reference(m.group(0)) or "XXXX",
        cleaned
    )

    # 2. Redact labeled account numbers, including space- or hyphen-grouped digits
    # e.g. 'Transfer to account 1234 5678 9012', 'A/C: 98765-43210', 'AC 123456789012'
    def _mask_labeled_account(m):
        label = m.group(1)
        raw_val = m.group(2)
        digits = re.sub(r'\D', '', raw_val)
        if len(digits) >= 5 or len(raw_val) >= 5:
            return f"{label} {mask_reference(raw_val) or 'XXXX'}"
        return m.group(0)

    # Order prefixes by descending length and enforce boundary \b after the label
    # Includes account labels and loan ID / loan reference labels
    account_prefix_pattern = r'(?i)\b(account|acc\.?\s*no|acc|a/c|ac|loan\s+(?:id|ref(?:erence)?)|ref(?:erence)?)\b\s*[:#-]?\s*'

    cleaned = re.sub(
        account_prefix_pattern + r'(\d+(?:[\s-]\d+){1,3})\b',
        _mask_labeled_account,
        cleaned
    )
    # Also handle single-token labeled account or loan ID
    cleaned = re.sub(
        account_prefix_pattern + r'([A-Za-z0-9]*\d[A-Za-z0-9]*)\b',
        lambda m: f"{m.group(1)} {mask_reference(m.group(2)) or 'XXXX'}",
        cleaned
    )

    # 3. Redact hyphen-separated long identifiers (e.g. '98765-43210-12', '1234-5678-9012')
    # where total digits >= 9. Done before phone check to avoid partial phone matches!
    def _mask_hyphen_seq(m):
        token = m.group(0)
        digits = re.sub(r'\D', '', token)
        if len(digits) >= 9:
            return mask_reference(digits) or "XXXX"
        return token

    cleaned = re.sub(
        r'\b\d{2,6}(?:-\d{2,6}){1,4}\b',
        _mask_hyphen_seq,
        cleaned
    )

    # 4. Redact formatted phone/mobile numbers with hyphens/spaces
    # e.g. 'Payment from Mob: 98765-43210', 'Mob 98765 43210'
    def _mask_phone(m):
        prefix = m.group(1) or ""
        digits = re.sub(r'\D', '', m.group(2))
        if len(digits) == 10 and digits[0] in "6789":
            prefix_str = f"{prefix} " if prefix else ""
            return f"{prefix_str}{mask_reference(digits) or 'XXXX'}"
        return m.group(0)

    cleaned = re.sub(
        r'(?i)\b(mob(?:ile)?|ph(?:one)?|tel)\s*[:#-]?\s*([6-9]\d{4}[\s-]\d{5})\b',
        _mask_phone,
        cleaned
    )
    # Standalone or prefixed 10-digit mobile numbers
    cleaned = re.sub(
        r'(?i)\b(mob(?:ile)?|ph(?:one)?|tel)?\s*[:#-]?\s*([6-9]\d{9})\b',
        lambda m: f"{m.group(1) + ' ' if m.group(1) else ''}{mask_reference(m.group(2)) or 'XXXX'}",
        cleaned
    )

    # 5. Redact contiguous sequences of 9 or more digits (card, account, or transaction references)
    cleaned = re.sub(
        r'\b\d{9,}\b',
        lambda m: mask_reference(m.group(0)) or "XXXX",
        cleaned
    )

    return cleaned.strip()


def sanitize_log_message(msg: str) -> str:
    """
    Helper to sanitize logs to prevent leaking raw account numbers or keys.
    """
    return sanitize_free_text(msg) or "REDACTED"
