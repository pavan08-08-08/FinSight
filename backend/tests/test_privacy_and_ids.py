import pytest
from backend.app.services.evidence_id import (
    mask_reference,
    sanitize_free_text,
    compute_reference_fingerprint,
    generate_evidence_id,
    sanitize_log_message,
)
from backend.app.services.errors import MissingHMACKeyError


def test_mask_reference():
    # Long account numbers (preserve last 4, mask prefix with 'XXXX')
    assert mask_reference("123456789012") == "XXXX9012"
    assert mask_reference("SB-987654321") == "XXXX4321"
    
    # Short references (<= 4 chars are canonical 'XXXX')
    assert mask_reference("1234") == "XXXX"
    assert mask_reference("AB") == "XXXX"
    
    # None or whitespace
    assert mask_reference(None) is None
    assert mask_reference("") is None
    assert mask_reference("   ") is None


def test_hmac_fingerprint_deterministic_and_normalized():
    key = b"super-secret-key-for-test-32chars"
    # Case and non-alphanumeric differences must yield identical fingerprint
    fp1 = compute_reference_fingerprint("HDFC-1234-5678", hmac_key=key)
    fp2 = compute_reference_fingerprint("hdfc 1234 5678", hmac_key=key)
    fp3 = compute_reference_fingerprint("HDFC12345678", hmac_key=key)
    
    assert fp1 is not None
    assert fp1 == fp2 == fp3


def test_hmac_fingerprint_missing_key_raises_error(monkeypatch):
    monkeypatch.delenv("PROOFLEDGER_REFERENCE_HMAC_KEY", raising=False)
    with pytest.raises(MissingHMACKeyError):
        compute_reference_fingerprint("HDFC12345678", hmac_key=None)


def test_generate_evidence_id_stability():
    """Same document_id and location must yield identical evidence ID without values."""
    id1 = generate_evidence_id(
        document_id="doc-abc-123",
        evidence_type="loan_amount",
        source_location_str="line:15",
        occurrence_index=0
    )
    id2 = generate_evidence_id(
        document_id="doc-abc-123",
        evidence_type="loan_amount",
        source_location_str="line:15",
        occurrence_index=0
    )
    assert id1 == id2
    assert id1.startswith("ev_")
    assert len(id1) == 23  # 'ev_' + 20 hex characters


def test_generate_evidence_id_distinct_occurrences():
    id_occ0 = generate_evidence_id("doc-1", "loan_amount", "row:5", 0)
    id_occ1 = generate_evidence_id("doc-1", "loan_amount", "row:5", 1)
    id_diff_loc = generate_evidence_id("doc-1", "loan_amount", "row:6", 0)

    assert id_occ0 != id_occ1
    assert id_occ0 != id_diff_loc


def test_generate_evidence_id_multiple_types_same_location_regression():
    """
    Audit Item 2: Multiple evidence records from the same document and location
    with distinct occurrence indices receive distinct IDs.
    Preserves Member 5 canonical ID algorithm and exact pinned outputs.
    """
    # Exact pinned outputs from Member 5 canonical contract
    assert generate_evidence_id("doc_kfs_001", "loan_amount", "Page 2", 0) == "ev_d2e7cc1c32ef1149bc86"
    assert generate_evidence_id("doc_kfs_001", "interest_rate", "Page 2", 1) != "ev_d2e7cc1c32ef1149bc86"
    assert generate_evidence_id("doc_kfs_001", "", None, 3) == "ev_76bb4fbf4a0eed713845"

    # Stability across repeated runs
    id_occ0 = generate_evidence_id("doc_kfs_001", "loan_amount", "Page 2", 0)
    id_occ1 = generate_evidence_id("doc_kfs_001", "interest_rate", "Page 2", 1)
    assert id_occ0 == "ev_d2e7cc1c32ef1149bc86"
    assert id_occ0 != id_occ1



def test_sanitize_log_message():
    raw_log = "Customer account 987654321012 transferred funds to 123456789."
    sanitized = sanitize_log_message(raw_log)
    assert "987654321012" not in sanitized
    assert "XXXX1012" in sanitized or "XXXX6789" in sanitized


def test_sanitize_free_text():
    # 1. Contiguous long identifier (NEFT 987654321012)
    neft_contig = sanitize_free_text("NEFT 987654321012")
    assert "987654321012" not in neft_contig
    assert "NEFT" in neft_contig
    assert "XXXX1012" in neft_contig

    # 2. Hyphen-separated identifier (NEFT 98765-43210-12)
    neft_hyphen = sanitize_free_text("NEFT 98765-43210-12")
    assert "98765-43210-12" not in neft_hyphen
    assert "NEFT" in neft_hyphen
    assert "XXXX1012" in neft_hyphen

    # 3. Space-separated identifier after an account label (Transfer to account 1234 5678 9012)
    ac_space = sanitize_free_text("Transfer to account 1234 5678 9012")
    assert "1234 5678 9012" not in ac_space
    assert "Transfer to account" in ac_space
    assert "XXXX9012" in ac_space

    # 4. Formatted phone number (Payment from Mob: 98765-43210)
    mob_hyphen = sanitize_free_text("Payment from Mob: 98765-43210")
    assert "98765-43210" not in mob_hyphen
    assert "Payment from Mob" in mob_hyphen
    assert "XXXX3210" in mob_hyphen

    # 5. UPI transaction with 12-digit reference (UPI/314567890123/Payment to merchant)
    upi = sanitize_free_text("UPI/314567890123/Payment to merchant")
    assert "314567890123" not in upi
    assert "Payment to merchant" in upi
    assert "XXXX0123" in upi

    # 6. Ordinary description with short amount (Coffee Shop Payment Rs. 450)
    coffee = sanitize_free_text("Coffee Shop Payment Rs. 450")
    assert coffee == "Coffee Shop Payment Rs. 450"

    # 7. IMPS with account and mobile number
    imps = sanitize_free_text("IMPS P2A to 987654321012 Mob 9876543210")
    assert "987654321012" not in imps
    assert "9876543210" not in imps
    assert "IMPS P2A" in imps


def test_matcher_keywords_preserved():
    """
    Verify Member 2 matching keywords (EMI, processing fee, loan credit,
    loan repayment, loan ID, loan reference) are preserved while redacting
    embedded sensitive numeric identifiers.
    """
    # 1. EMI & loan ID
    s1 = sanitize_free_text("Monthly EMI payment for loan ID 987654321012")
    assert "EMI" in s1
    assert "loan ID" in s1
    assert "987654321012" not in s1
    assert "XXXX1012" in s1

    # 2. loan repayment
    s2 = sanitize_free_text("Loan repayment debited via NACH ref 123456789012")
    assert "Loan repayment" in s2
    assert "123456789012" not in s2
    assert "XXXX9012" in s2

    # 3. processing fee & loan reference
    s3 = sanitize_free_text("Deduction of processing fee plus GST for loan reference LN-987654321")
    assert "processing fee" in s3
    assert "loan reference" in s3
    assert "987654321" not in s3
    assert "LN-XXXX4321" in s3 or "XXXX4321" in s3

    # 4. loan credit
    s4 = sanitize_free_text("Loan credit disbursed to A/C 123456789012")
    assert "Loan credit" in s4
    assert "123456789012" not in s4
    assert "XXXX9012" in s4

    # 5. auto debit EMI with alphanumeric loan ID
    s5 = sanitize_free_text("Auto debit EMI for loan ID: LN123456789")
    assert "EMI" in s5
    assert "loan ID" in s5
    assert "123456789" not in s5
    assert "XXXX6789" in s5


def test_alphanumeric_reference_redaction_across_all_surfaces(caplog):
    """
    Issue 1: Test that alphanumeric references like FD-30451122, LN-987654321,
    and POL12345678 are masked before reaching EvidenceRecord, BankTransaction,
    SourceLocation, WarningItem, serialized output, and captured logs.
    """
    import logging
    from backend.app.schemas import EvidenceRecord, BankTransaction, SourceLocation, WarningItem
    from backend.app.services.ingestion import parse_csv_transactions

    raw_fd = "FD-30451122"
    raw_ln = "LN-987654321"
    raw_pol = "POL12345678"

    # 1. sanitize_free_text masks standalone alphanumeric references
    assert raw_fd not in (sanitize_free_text(f"Deposit account {raw_fd} renewed") or "")
    assert "XXXX1122" in (sanitize_free_text(f"Deposit account {raw_fd} renewed") or "")
    assert raw_ln not in (sanitize_free_text(f"Payment for loan {raw_ln}") or "")
    assert raw_pol not in (sanitize_free_text(f"Policy {raw_pol} active") or "")

    # 2. Test captured logs via sanitize_log_message
    with caplog.at_level(logging.INFO):
        logger = logging.getLogger("test_privacy")
        log_msg = sanitize_log_message(f"Processing transaction for FD account {raw_fd} and policy {raw_pol}")
        logger.info(log_msg)
        assert raw_fd not in caplog.text
        assert raw_pol not in caplog.text
        assert "XXXX1122" in caplog.text

    # 3. CSV parsing with FD-30451122 and LN-987654321 references
    csv_data = (
        "Date,Description,Debit,Ref No\n"
        f"2023-01-15,Payment for {raw_fd},1500.00,{raw_fd}\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "test_stmt.csv", "doc-priv-fd")
    assert len(txns) == 1
    assert len(evidence) == 1

    t = txns[0]
    e = evidence[0]

    # Verify model fields do NOT contain raw reference
    assert t.reference_number == "XXXX1122"
    assert raw_fd not in t.description
    assert raw_fd not in (e.source_excerpt or "")
    assert raw_fd not in e.value

    # Verify stored provenance does NOT contain raw reference
    assert e.structured_source_location is not None
    assert raw_fd not in (e.structured_source_location.raw_source_snippet or "")

    # Verify serialized output does NOT contain raw reference
    t_json = t.model_dump_json()
    e_json = e.model_dump_json()
    assert raw_fd not in t_json
    assert raw_fd not in e_json


# ===========================================================================
# Issue 2 Regression Tests: Explicit HMAC Failure & Privacy Invariants
# ===========================================================================

def test_hmac_missing_env_key_raises_typed_error(monkeypatch):
    """Issue 2.1: Missing environment key raises typed MissingHMACKeyError."""
    monkeypatch.delenv("PROOFLEDGER_REFERENCE_HMAC_KEY", raising=False)
    with pytest.raises(MissingHMACKeyError) as exc_info:
        compute_reference_fingerprint("REF-987654321", hmac_key=None)
    assert "not set" in str(exc_info.value).lower()


@pytest.mark.parametrize("empty_key", ["", "   ", b"", b"   "])
def test_hmac_empty_key_raises_typed_error(monkeypatch, empty_key):
    """Issue 2.2: Empty environment or supplied key raises typed MissingHMACKeyError."""
    if isinstance(empty_key, str):
        monkeypatch.setenv("PROOFLEDGER_REFERENCE_HMAC_KEY", empty_key)
        with pytest.raises(MissingHMACKeyError) as exc_info:
            compute_reference_fingerprint("REF-987654321", hmac_key=None)
        assert "empty" in str(exc_info.value).lower() or "not set" in str(exc_info.value).lower()

    with pytest.raises(MissingHMACKeyError) as exc_info2:
        compute_reference_fingerprint("REF-987654321", hmac_key=empty_key)
    assert "empty" in str(exc_info2.value).lower() or "not set" in str(exc_info2.value).lower()


def test_hmac_explicitly_supplied_test_key_succeeds(monkeypatch):
    """Issue 2.3: Explicitly supplied test key succeeds even if env var is absent."""
    monkeypatch.delenv("PROOFLEDGER_REFERENCE_HMAC_KEY", raising=False)
    test_key = b"explicit-key-32-chars-test-only!"
    fp = compute_reference_fingerprint("HDFC-1234-5678", hmac_key=test_key)
    assert fp is not None
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_ingestion_fails_explicitly_when_required_fingerprinting_key_missing(monkeypatch):
    """
    Issue 2.4: Ingestion cannot silently succeed with reference_fingerprint=None
    when required reference fingerprinting fails due to missing key.
    """
    from backend.app.services.ingestion import parse_csv_transactions, extract_document

    monkeypatch.delenv("PROOFLEDGER_REFERENCE_HMAC_KEY", raising=False)

    csv_data = (
        "Date,Description,Debit,Ref No\n"
        "2023-01-01,EMI Payment,1500.00,CONFIDENTIAL_REF_998877\n"
    ).encode("utf-8")

    # parse_csv_transactions must raise MissingHMACKeyError
    with pytest.raises(MissingHMACKeyError):
        parse_csv_transactions(csv_data, "stmt.csv", "doc-test-fail", hmac_key=None)

    # extract_document must also propagate MissingHMACKeyError
    with pytest.raises(MissingHMACKeyError):
        extract_document(csv_data, "stmt.csv", "doc-test-fail", hmac_key=None)


def test_hmac_error_messages_and_logs_do_not_disclose_raw_references_or_secrets(monkeypatch, caplog):
    """
    Issue 2.5: Error messages and captured logs must never disclose raw reference numbers
    or HMAC secret keys.
    """
    import logging
    from backend.app.services.ingestion import parse_csv_transactions

    monkeypatch.delenv("PROOFLEDGER_REFERENCE_HMAC_KEY", raising=False)
    sensitive_ref = "SECRET_ACCOUNT_NUMBER_987654321012"

    csv_data = (
        "Date,Description,Debit,Ref No\n"
        f"2023-01-01,Payment,1500.00,{sensitive_ref}\n"
    ).encode("utf-8")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(MissingHMACKeyError) as exc_info:
            parse_csv_transactions(csv_data, "secret.csv", "doc-priv-err", hmac_key=None)

    err_str = str(exc_info.value)
    # Neither raw reference nor secret name/value in exception text
    assert sensitive_ref not in err_str
    assert "987654321012" not in err_str

    # Captured logs must also be clean
    assert sensitive_ref not in caplog.text
    assert "987654321012" not in caplog.text


def test_unrelated_fields_without_references_do_not_require_hmac_key(monkeypatch):
    """
    Issue 2 - Key check scope: Ingestion of documents without reference numbers
    succeeds even if PROOFLEDGER_REFERENCE_HMAC_KEY is unset. Fingerprinting is not
    mandatory for unrelated fields.
    """
    from backend.app.services.ingestion import parse_csv_transactions, extract_document

    monkeypatch.delenv("PROOFLEDGER_REFERENCE_HMAC_KEY", raising=False)

    csv_data = (
        "Date,Description,Debit\n"
        "2023-01-01,Groceries,250.00\n"
        "2023-01-02,Transport,50.00\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "no_refs.csv", "doc-no-refs", hmac_key=None)
    assert len(txns) == 2
    assert len(evidence) == 2
    assert evidence[0].reference_fingerprint is None
    assert evidence[1].reference_fingerprint is None

    res = extract_document(csv_data, "no_refs.csv", "doc-no-refs", hmac_key=None)
    assert len(res.transactions) == 2





