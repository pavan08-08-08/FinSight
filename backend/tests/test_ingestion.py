import datetime as dt
from decimal import Decimal
import pytest
from backend.app.schemas import (
    SourceLocation,
    EvidenceRecord,
    BankTransaction,
    EvidenceType,
    TransactionType,
    VerificationStatus,
    WarningCodes,
    ExtractionResult,
)
from backend.app.services.ingestion import (
    validate_file_upload,
    parse_csv_transactions,
    parse_txt_document,
    parse_pdf_document,
    extract_document,
    verify_evidence,
)
from backend.app.services.errors import (
    FileSizeLimitExceededError,
    UnsupportedFileTypeError,
    CorruptedFileError,
    EncryptedPDFError,
    EmptyDocumentError,
    UnsupportedScannedPDFError,
    MissingHMACKeyError,
)


def test_validate_file_upload_valid_files():
    # Valid TXT
    res = validate_file_upload(b"Some text", "doc.txt")
    assert res.file_type == "txt"
    assert res.is_valid is True

    # Valid CSV
    res = validate_file_upload(b"Date,Amount\n2023-01-01,100", "statement.csv")
    assert res.file_type == "csv"
    assert res.is_valid is True

    # Valid PDF magic bytes
    res = validate_file_upload(b"%PDF-1.4 binary data", "doc.pdf")
    assert res.file_type == "pdf"
    assert res.is_valid is True


def test_validate_file_upload_rejects_oversized_file():
    eleven_mb = b"x" * (11 * 1024 * 1024)
    with pytest.raises(FileSizeLimitExceededError):
        validate_file_upload(eleven_mb, "large_statement.csv")


def test_validate_file_upload_rejects_unsupported_extensions():
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_upload(b"print('hello')", "script.py")


def test_validate_file_upload_rejects_fake_pdf():
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_upload(b"NOT A REAL PDF", "statement.pdf")


def test_parse_csv_transactions_separate_debit_credit():
    csv_data = (
        "Txn Date,Particulars,Debit,Credit,Balance,Ref No\n"
        "2023-04-10,Salary Credited,,50000.00,50000.00,SAL123456\n"
        "2023-04-12,Loan EMI Payment,12500.50,,37499.50,EMI987654\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(
        file_content=csv_data,
        filename="bank_statement.csv",
        document_id="doc-test-111"
    )

    assert len(txns) == 2
    assert len(evidence) == 2

    # Verify first transaction (Credit)
    assert txns[0].type == TransactionType.CREDIT
    assert txns[0].amount == Decimal("50000.00")
    assert txns[0].date == dt.date(2023, 4, 10)
    assert txns[0].balance == Decimal("50000.00")
    assert txns[0].reference_number == "XXXX3456"

    # Verify second transaction (Debit)
    assert txns[1].type == TransactionType.DEBIT
    assert txns[1].amount == Decimal("12500.50")
    assert txns[1].reference_number == "XXXX7654"

    # Verify evidence provenance
    loc0 = SourceLocation.from_canonical_string(evidence[0].source_location)
    loc1 = SourceLocation.from_canonical_string(evidence[1].source_location)
    assert loc0.row_index == 1
    assert loc1.row_index == 2
    assert evidence[0].verification_status == "extracted"


def test_parse_csv_transactions_single_amount_and_type_col():
    csv_data = (
        "Date,Narration,Amount,Type\n"
        "15-Jan-2023,ATM Withdrawal,2000.00,DR\n"
        "18-Jan-2023,UPI Received,500.00,CR\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(
        file_content=csv_data,
        filename="bank_statement_2.csv",
        document_id="doc-test-222"
    )

    assert len(txns) == 2
    assert txns[0].type == TransactionType.DEBIT
    assert txns[0].amount == Decimal("2000.00")
    assert txns[1].type == TransactionType.CREDIT
    assert txns[1].amount == Decimal("500.00")


def test_parse_csv_transactions_handles_missing_and_malformed_rows():
    csv_data = (
        "Date,Description,Debit,Credit\n"
        ",,,\n"  # Blank row
        "Invalid Date,Broken row,not_a_number,\n"  # Malformed amount row (skipped)
        "2023-05-01,Valid row,1000.00,\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(
        file_content=csv_data,
        filename="messy.csv",
        document_id="doc-test-333"
    )

    assert len(txns) == 1
    assert txns[0].amount == Decimal("1000.00")
    assert txns[0].type == TransactionType.DEBIT


def test_parse_txt_document_kfs_mentions():
    txt_data = (
        "Key Fact Statement (KFS)\n"
        "Loan Amount: Rs. 2,00,000.00\n"
        "Interest Rate: 11.5% p.a.\n"
        "Tenure: 24 months\n"
    ).encode("utf-8")

    raw_text, evidence = parse_txt_document(
        file_content=txt_data,
        filename="kfs_notice.txt",
        document_id="doc-kfs-444"
    )

    assert len(evidence) == 2
    types = [e.type for e in evidence]
    assert EvidenceType.LOAN_AMOUNT in types
    assert EvidenceType.INTEREST_RATE in types

    loan_ev = next(e for e in evidence if e.type == EvidenceType.LOAN_AMOUNT)
    assert loan_ev.normalized_value == "200000.00"
    loc = SourceLocation.from_canonical_string(loan_ev.source_location)
    assert loc.line_number == 2
    assert loan_ev.verification_status == "extracted"


def test_parse_csv_raw_source_snippet_privacy():
    """Verify that CSV evidence snippets do not leak unmasked account or reference numbers."""
    csv_data = (
        "Date,Narration,Amount,Ref No\n"
        "2023-03-01,Transfer to Account 987654321012,5000.00,REF99887766\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(
        file_content=csv_data,
        filename="sensitive.csv",
        document_id="doc-priv-1"
    )

    assert len(evidence) == 1
    ev = evidence[0]

    # Verify structured SourceLocation adapter has row index
    loc = SourceLocation.from_canonical_string(ev.source_location)
    assert loc.row_index == 1

    # Raw excerpt must NOT contain the unmasked account number or raw joined CSV row
    assert "987654321012" not in (ev.source_excerpt or "")
    assert "REF99887766" not in (ev.source_excerpt or "")
    assert "Transfer to Account 987654321012,5000.00,REF99887766" not in (ev.source_excerpt or "")

    # Reference number must be masked in evidence and transaction
    assert ev.reference_number == "XXXX7766"
    assert txns[0].reference_number == "XXXX7766"

    # NormalizedTransaction.description must also be sanitized
    assert "987654321012" not in txns[0].description
    assert "Transfer to Account" in txns[0].description
    assert "XXXX1012" in txns[0].description


def test_parse_csv_matcher_preservation_and_direction():
    """
    Verify that CSV ingestion preserves matcher keywords, correctly assigns direction
    and positive amount, uses sanitized narration in both description and source_excerpt,
    and emits verification_status='extracted' (never auto-confirmed).
    """
    csv_data = (
        "Date,Narration,Debit,Credit,Ref No\n"
        "2023-06-01,Monthly EMI payment for loan ID 987654321012,15000.00,,LNREF123456\n"
        "2023-06-02,Loan credit disbursed to A/C 987654321012,,200000.00,DISB987654\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(
        file_content=csv_data,
        filename="bank_stmt_matching.csv",
        document_id="doc-match-1"
    )

    assert len(txns) == 2
    assert len(evidence) == 2

    # Row 1: EMI Debit
    t1 = txns[0]
    e1 = evidence[0]
    assert t1.type == TransactionType.DEBIT
    assert t1.amount == Decimal("15000.00")  # Positive Decimal
    assert "EMI" in t1.description
    assert "loan ID" in t1.description
    assert "987654321012" not in t1.description
    assert "XXXX1012" in t1.description
    assert t1.reference_number == "XXXX3456"
    assert e1.source_excerpt == f"Row 1: {t1.description}"
    assert e1.verification_status == "extracted"  # Member 1 does NOT confirm matches!

    # Row 2: Loan Credit
    t2 = txns[1]
    e2 = evidence[1]
    assert t2.type == TransactionType.CREDIT
    assert t2.amount == Decimal("200000.00")  # Positive Decimal
    assert "Loan credit" in t2.description
    assert "987654321012" not in t2.description
    assert "XXXX1012" in t2.description
    assert t2.reference_number == "XXXX7654"
    assert e2.source_excerpt == f"Row 2: {t2.description}"
    assert e2.verification_status == "extracted"


def test_structured_provenance_survives_internally_until_serialization():
    """
    Audit Item 1: Verify whether raw_source_snippet, page/row/line information,
    and character offsets survive internally on EvidenceRecord.structured_source_location,
    while canonical EvidenceRecord.source_location remains a string.
    """
    # 1. Verify CSV transaction evidence record structured provenance
    csv_data = "Date,Narration,Debit,Ref No\n2023-01-01,EMI Payment,1500.00,REF123456\n".encode("utf-8")
    txns, evidence = parse_csv_transactions(csv_data, "stmt.csv", "doc-csv-prov")
    assert len(evidence) == 1
    ev_csv = evidence[0]

    # Shared canonical schema boundary check: source_location must be str
    assert isinstance(ev_csv.source_location, str)
    assert ev_csv.source_location == "row 1"

    # Internal structured provenance survival
    loc_csv = ev_csv.structured_source_location
    assert loc_csv is not None
    assert isinstance(loc_csv, SourceLocation)
    assert loc_csv.row_index == 1
    assert "EMI Payment" in (loc_csv.raw_source_snippet or "")
    # Sensitive reference must NOT appear unmasked in raw_source_snippet
    assert "123456" not in (loc_csv.raw_source_snippet or "")

    # 2. Verify TXT parse document structured provenance with line and char offsets
    txt_data = "Key Fact Statement\nLoan Amount: Rs. 50,000.00\n".encode("utf-8")
    _, txt_evidence = parse_txt_document(txt_data, "kfs.txt", "doc-txt-prov")
    assert len(txt_evidence) == 1
    ev_txt = txt_evidence[0]

    assert isinstance(ev_txt.source_location, str)
    assert ev_txt.source_location == "line 2"

    loc_txt = ev_txt.structured_source_location
    assert loc_txt is not None
    assert isinstance(loc_txt, SourceLocation)
    assert loc_txt.line_number == 2
    assert loc_txt.char_start is not None and loc_txt.char_end is not None
    assert "Loan Amount" in (loc_txt.raw_source_snippet or "")

    # 3. Verify serialization boundary does NOT expose internal private fields
    dumped = ev_txt.model_dump()
    assert isinstance(dumped["source_location"], str)
    assert "_source_location_obj" not in dumped
    assert "structured_source_location" not in dumped


def test_multiple_evidence_same_location_no_collision():
    """
    Audit Item 2: Verify that multiple evidence items found at the exact same location
    (e.g., both loan amount and interest rate on the same line) receive distinct,
    deterministic occurrence indices and do NOT collide.
    """
    txt_data = "Loan Amount: Rs. 75,000.00 and Interest Rate: 10.5% p.a.\n".encode("utf-8")
    _, evidence = parse_txt_document(txt_data, "combined_kfs.txt", "doc-comb-1")

    assert len(evidence) == 2
    ev_loan = next(e for e in evidence if e.type == EvidenceType.LOAN_AMOUNT)
    ev_rate = next(e for e in evidence if e.type == EvidenceType.INTEREST_RATE)

    # Both are on line 1
    assert ev_loan.source_location == "line 1"
    assert ev_rate.source_location == "line 1"

    # But their evidence IDs must be distinct!
    assert ev_loan.id != ev_rate.id
    assert ev_loan.id.startswith("ev_")
    assert ev_rate.id.startswith("ev_")

    # Re-running parsing produces the exact same deterministic IDs
    _, evidence_again = parse_txt_document(txt_data, "combined_kfs.txt", "doc-comb-1")
    assert evidence_again[0].id == evidence[0].id
    assert evidence_again[1].id == evidence[1].id


# ===========================================================================
# Hardening Regression Tests: Issues 2, 3, 4, 5
# ===========================================================================


def test_csv_direction_explicit_columns_and_type():
    """
    Issue 2: Verify explicit credit/debit columns, explicit type column,
    and negative signed amount convention.
    """
    csv_data = (
        "Date,Description,Amount,Type\n"
        "2023-02-01,Service Charge,150.00,DR\n"
        "2023-02-02,Interest Credit,50.00,CR\n"
        "2023-02-03,Negative signed debit,-200.00,\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "type_test.csv", "doc-type-1")
    assert len(txns) == 3
    assert txns[0].type == TransactionType.DEBIT
    assert txns[0].amount == Decimal("150.00")
    assert txns[1].type == TransactionType.CREDIT
    assert txns[1].amount == Decimal("50.00")
    assert txns[2].type == TransactionType.DEBIT
    assert txns[2].amount == Decimal("200.00")


def test_csv_direction_ambiguous_positive_amount_emits_warning():
    """
    Issue 2: A positive amount alone without explicit direction or direction keywords
    must NOT be assumed to be a credit. It must emit an AMBIGUOUS_COLUMN warning.
    """
    csv_data = (
        "Date,Description,Amount\n"
        "2023-02-05,Misc reference 8877,500.00\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "ambig_dir.csv", "doc-ambig-1")
    assert len(txns) == 0  # Does NOT construct misleading BankTransaction!
    assert len(evidence) == 1
    ev = evidence[0]
    assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in ev.warnings)
    assert any("Ambiguous transaction direction" in w.message for w in ev.warnings)


def test_csv_direction_contradictory_combinations():
    """
    Issue 2: Detect contradictory direction combinations:
    1) Both debit and credit columns have amounts.
    2) Negative amount with explicit credit type indicator.
    """
    csv_data = (
        "Date,Description,Debit,Credit\n"
        "2023-03-01,Both debit and credit,100.00,200.00\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "contradict1.csv", "doc-contra-1")
    assert len(txns) == 0
    assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in evidence[0].warnings)

    csv_data2 = (
        "Date,Description,Amount,Type\n"
        "2023-03-02,Contradictory negative credit,-500.00,CR\n"
    ).encode("utf-8")

    txns2, evidence2 = parse_csv_transactions(csv_data2, "contradict2.csv", "doc-contra-2")
    assert len(txns2) == 0
    assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in evidence2[0].warnings)


def test_csv_negative_balance_preserved_and_not_coerced():
    """
    Issue 3: Balance may be negative (overdraft). Do not convert negative balance
    to absolute value; preserve negative sign exactly.
    """
    csv_data = (
        "Date,Description,Debit,Credit,Balance\n"
        "2023-04-01,Overdraft fee,500.00,,-1200.50\n"
        "2023-04-02,Salary credit,,3000.00,1799.50\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "balance.csv", "doc-bal-1")
    assert len(txns) == 2
    assert txns[0].balance == Decimal("-1200.50")
    assert txns[1].balance == Decimal("1799.50")


def test_csv_zero_amount_and_malformed_amounts_rejected():
    """
    Issue 3 & 4: Zero amounts and malformed numeric values must be rejected
    with structured VALUE_NOT_FOUND warnings, not silently coerced.
    """
    csv_data = (
        "Date,Description,Debit,Credit\n"
        "2023-05-01,Zero amount row,0.00,\n"
        "2023-05-02,Corrupt amount row,corrupted_val,\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "zero_corrupt.csv", "doc-zero-1")
    assert len(txns) == 0
    assert len(evidence) == 2
    assert any(w.code == WarningCodes.VALUE_NOT_FOUND for w in evidence[0].warnings)
    assert any(w.code == WarningCodes.VALUE_NOT_FOUND for w in evidence[1].warnings)


def test_extract_document_emits_extraction_result_with_warnings():
    """
    Issue 4: extract_document produces a valid ExtractionResult aggregating
    all transactions, evidence, and structured warnings without leakage.
    """
    csv_data = (
        "Date,Description,Debit,Credit\n"
        "2023-06-01,Valid transaction,1000.00,\n"
        "05/06/2023,Ambiguous date row,500.00,\n"
    ).encode("utf-8")

    res = extract_document(csv_data, "stmt_warnings.csv", "doc-ext-1")
    assert isinstance(res, ExtractionResult)
    assert res.file_type == "csv"
    assert len(res.transactions) == 1
    assert len(res.evidence) == 2
    assert len(res.warnings) >= 1
    assert any(w.code == WarningCodes.AMBIGUOUS_DATE for w in res.warnings)


def test_verify_evidence_requires_source_and_updates_status():
    """
    Issue 5: Verification status must be justified.
    - Transition to verified without verification source raises ValueError.
    - Conflicting evidence cannot transition to verified.
    - Valid verification records verification source and updates status.
    - Newly extracted evidence stays 'extracted'.
    """
    csv_data = "Date,Description,Debit\n2023-07-01,Payment,500.00\n".encode("utf-8")
    txns, evidence = parse_csv_transactions(csv_data, "stmt.csv", "doc-ver-1")
    assert len(evidence) == 1
    ev = evidence[0]

    # Ingestion alone leaves status as 'extracted'
    assert ev.verification_status == VerificationStatus.EXTRACTED

    # Verification without source is rejected
    with pytest.raises(ValueError, match="Verification source is required"):
        verify_evidence(ev, verification_source="")

    with pytest.raises(ValueError, match="Verification source is required"):
        verify_evidence(ev, verification_source="   ")

    # Valid verification succeeds
    verified_ev = verify_evidence(
        ev,
        verification_source="Bank Statement cross-referenced with Loan Agreement LN-987654321",
        notes="Audit confirmed on 2023-07-05"
    )
    assert verified_ev.verification_status == VerificationStatus.VERIFIED
    assert "Verified via:" in (verified_ev.confidence_basis or "")
    # Sensitive reference LN-987654321 must be masked in confidence_basis!
    assert "987654321" not in verified_ev.confidence_basis
    assert "XXXX4321" in verified_ev.confidence_basis

    # Conflicting evidence cannot be marked verified
    conflicting_ev = EvidenceRecord(
        id="ev_conflict",
        document_id="doc-conf",
        type=EvidenceType.LOAN_AMOUNT,
        value="50000",
        source_document="kfs.txt",
        source_location="line 1",
        verification_status=VerificationStatus.CONFLICTING
    )
    with pytest.raises(ValueError, match="Conflicting evidence cannot be marked as verified"):
        verify_evidence(conflicting_ev, verification_source="Independent review")


# ===========================================================================
# Issue 1 Regression Tests: Missing, Blank, Malformed, Zero Transaction Amounts
# ===========================================================================

def test_csv_missing_amount_column_rejects_all_rows_with_warning():
    """
    Issue 1.1: CSV with no debit, credit, or amount columns rejects every row.
    - Emits structured VALUE_NOT_FOUND warning with safe row provenance.
    - Does NOT fabricate BankTransaction or 'TRANSACTION 0.00' evidence.
    - Does NOT expose raw account numbers in warning message or provenance.
    - extract_document preserves warnings.
    """
    csv_data = (
        "Date,Particulars\n"
        "2023-01-01,Transfer to Account 123456789012\n"
        "2023-01-02,ATM Cash Withdrawal\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "no_amount_col.csv", "doc-no-amt")
    assert len(txns) == 0
    assert len(evidence) == 2

    for ev in evidence:
        assert ev.amount is None
        assert ev.normalized_value is None
        assert ev.value != "TRANSACTION 0.00"
        assert "0.00" not in ev.value
        assert ev.type == EvidenceType.FINANCIAL_MENTION
        assert any(w.code == WarningCodes.VALUE_NOT_FOUND for w in ev.warnings)
        # Raw account number must NOT leak into warning message or provenance
        for w in ev.warnings:
            assert "123456789012" not in w.message
        assert "123456789012" not in (ev.source_excerpt or "")

    # extract_document aggregation test
    res = extract_document(csv_data, "no_amount_col.csv", "doc-no-amt")
    assert len(res.transactions) == 0
    assert len(res.warnings) >= 1
    assert any(w.code == WarningCodes.VALUE_NOT_FOUND for w in res.warnings)


def test_csv_blank_and_whitespace_amount_cells_rejected():
    """
    Issue 1.2 & 1.3: Blank and whitespace-only amount cells are rejected with
    structured VALUE_NOT_FOUND warnings and produce no BankTransaction.
    """
    csv_data = (
        "Date,Description,Amount,Type\n"
        "2023-02-01,Blank amount cell,,DR\n"
        "2023-02-02,Whitespace amount cell,   ,DR\n"
        "2023-02-03,Tab whitespace cell,\t\t,CR\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "blank_amt.csv", "doc-blank-amt")
    assert len(txns) == 0
    assert len(evidence) == 3

    for ev in evidence:
        assert ev.amount is None
        assert ev.normalized_value is None
        assert ev.value != "TRANSACTION 0.00"
        assert ev.type == EvidenceType.FINANCIAL_MENTION
        assert any(w.code == WarningCodes.VALUE_NOT_FOUND for w in ev.warnings)


def test_csv_malformed_amounts_rejected():
    """
    Issue 1.4: Malformed amount strings are rejected with VALUE_NOT_FOUND warnings.
    """
    csv_data = (
        "Date,Description,Amount,Type\n"
        "2023-03-01,Alphabetical amount,not_a_number,DR\n"
        "2023-03-02,Double negative amount,--500.00,DR\n"
        "2023-03-03,Multiple dots in amount,12.34.56,CR\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "malformed_amt.csv", "doc-malformed-amt")
    assert len(txns) == 0
    assert len(evidence) == 3

    for ev in evidence:
        assert ev.amount is None
        assert ev.normalized_value is None
        assert ev.value != "TRANSACTION 0.00"
        assert ev.type == EvidenceType.FINANCIAL_MENTION
        assert any(w.code == WarningCodes.VALUE_NOT_FOUND for w in ev.warnings)


def test_csv_zero_amount_not_fabricated_as_transaction():
    """
    Issue 1.5: Zero amounts must not become valid transactions because canonical
    BankTransaction requires amount > 0. Must not fabricate 'TRANSACTION 0.00'.
    """
    csv_data = (
        "Date,Description,Debit,Credit\n"
        "2023-04-01,Zero debit,0.00,\n"
        "2023-04-02,Zero credit,,0.00\n"
        "2023-04-03,Both zero,0.00,0.00\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "zeros.csv", "doc-zeros")
    assert len(txns) == 0
    assert len(evidence) == 3

    for ev in evidence:
        assert ev.amount is None
        assert ev.normalized_value is None
        assert ev.value != "TRANSACTION 0.00"
        assert ev.type == EvidenceType.FINANCIAL_MENTION
        assert any(w.code == WarningCodes.VALUE_NOT_FOUND for w in ev.warnings)


def test_csv_valid_rows_parse_correctly_alongside_rejected_rows():
    """
    Issue 1.6 & 1.7: Valid rows in the same CSV still parse correctly into BankTransactions
    while malformed, blank, zero, and missing rows in the same CSV are rejected with warnings.
    """
    csv_data = (
        "Date,Description,Amount,Type\n"
        "2023-05-01,Valid first row,100.00,DR\n"
        "2023-05-02,Blank amount row,,DR\n"
        "2023-05-03,Whitespace amount row,   ,DR\n"
        "2023-05-04,Malformed amount row,corrupted_val,CR\n"
        "2023-05-05,Zero amount row,0.00,CR\n"
        "2023-05-06,Valid last row,2500.50,CR\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "mixed.csv", "doc-mixed")
    assert len(txns) == 2
    assert len(evidence) == 6

    # Valid row 1
    assert txns[0].amount == Decimal("100.00")
    assert txns[0].type == TransactionType.DEBIT
    assert evidence[0].type == EvidenceType.REPAYMENT_TRANSACTION
    assert evidence[0].amount == Decimal("100.00")

    # Valid row 6
    assert txns[1].amount == Decimal("2500.50")
    assert txns[1].type == TransactionType.CREDIT
    assert evidence[5].type == EvidenceType.REPAYMENT_TRANSACTION
    assert evidence[5].amount == Decimal("2500.50")

    # Rejected rows 2, 3, 4, 5
    for idx in [1, 2, 3, 4]:
        assert evidence[idx].amount is None
        assert evidence[idx].type == EvidenceType.FINANCIAL_MENTION
        assert any(w.code == WarningCodes.VALUE_NOT_FOUND for w in evidence[idx].warnings)

    # Aggregated extraction result check
    res = extract_document(csv_data, "mixed.csv", "doc-mixed")
    assert len(res.transactions) == 2
    assert len(res.evidence) == 6
    assert len(res.warnings) >= 4


# ===========================================================================
# Issue 3 Regression Tests: Contradictory & Ambiguous Transaction Directions
# ===========================================================================

def test_csv_both_debit_and_credit_columns_populated_with_and_without_amount_col():
    """
    Issue 3.1: Rows with both debit and credit columns populated with non-zero
    amounts must be rejected, both without and with a separate amount column.
    A separate amount column must not override contradictory debit/credit indicators.
    """
    # 1. Without separate amount column
    csv_without_amt = (
        "Date,Description,Debit,Credit\n"
        "2023-06-01,Both debit and credit non-zero,150.00,300.00\n"
    ).encode("utf-8")
    txns1, ev1 = parse_csv_transactions(csv_without_amt, "contra_no_amt.csv", "doc-c1")
    assert len(txns1) == 0
    assert len(ev1) == 1
    assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in ev1[0].warnings)
    assert any("both debit and credit" in w.message for w in ev1[0].warnings)

    # 2. With separate amount column (must NOT override the contradiction!)
    csv_with_amt = (
        "Date,Description,Debit,Credit,Amount\n"
        "2023-06-02,Both cols plus amount col,150.00,300.00,150.00\n"
    ).encode("utf-8")
    txns2, ev2 = parse_csv_transactions(csv_with_amt, "contra_with_amt.csv", "doc-c2")
    assert len(txns2) == 0
    assert len(ev2) == 1
    assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in ev2[0].warnings)
    assert any("both debit and credit" in w.message for w in ev2[0].warnings)


def test_csv_negative_debit_column_value_rejected():
    """
    Issue 3.2: Negative value in explicit debit column must be checked for contradiction
    and rejected. Do not silently take the absolute value.
    """
    csv_data = (
        "Date,Description,Debit,Credit\n"
        "2023-06-03,Negative debit value,-250.00,\n"
        "2023-06-04,Parenthesized negative debit,(350.00),\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "neg_debit.csv", "doc-neg-d")
    assert len(txns) == 0
    assert len(evidence) == 2
    for ev in evidence:
        assert ev.amount is None
        assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in ev.warnings)
        assert any("negative value in debit column" in w.message.lower() for w in ev.warnings)


def test_csv_negative_credit_column_value_rejected():
    """
    Issue 3.3: Negative value in explicit credit column must be checked for contradiction
    and rejected. Do not silently take the absolute value.
    """
    csv_data = (
        "Date,Description,Debit,Credit\n"
        "2023-06-05,Negative credit value,,-500.00\n"
        "2023-06-06,Parenthesized negative credit,,(750.00)\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "neg_credit.csv", "doc-neg-c")
    assert len(txns) == 0
    assert len(evidence) == 2
    for ev in evidence:
        assert ev.amount is None
        assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in ev.warnings)
        assert any("negative value in credit column" in w.message.lower() for w in ev.warnings)


def test_csv_negative_amount_paired_with_explicit_cr_rejected():
    """
    Issue 3.4: Negative signed amount paired with explicit CR type indicator
    contradicts direction and must be rejected.
    """
    csv_data = (
        "Date,Description,Amount,Type\n"
        "2023-06-07,Negative amount with CR,-1200.00,CR\n"
        "2023-06-08,Parenthesized amount with Credit,(800.00),Credit\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "neg_cr.csv", "doc-neg-cr")
    assert len(txns) == 0
    assert len(evidence) == 2
    for ev in evidence:
        assert ev.amount is None
        assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in ev.warnings)
        assert any("negative amount" in w.message.lower() for w in ev.warnings)


def test_csv_positive_amount_paired_with_conflicting_dr_cr_indicators_rejected():
    """
    Issue 3.5: Positive amount paired with conflicting DR and CR indicators
    must be rejected.
    """
    csv_data = (
        "Date,Description,Amount,Type\n"
        "2023-06-09,Conflicting DR/CR type,500.00,DR/CR\n"
        "2023-06-10,Conflicting CR/DR type,600.00,CR/DR\n"
        "2023-06-11,Conflicting DR CR spaced,700.00,DR CR\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "conflict_type.csv", "doc-conf-type")
    assert len(txns) == 0
    assert len(evidence) == 3
    for ev in evidence:
        assert ev.amount is None
        assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in ev.warnings)
        assert any("conflicting explicit type" in w.message.lower() for w in ev.warnings)


def test_csv_contradictory_type_and_column_indicators_rejected():
    """
    Issue 3.6: Contradictory type indicator and column indicators (e.g. Debit column
    populated but Type is CR, or Credit column populated but Type is DR).
    """
    csv_data = (
        "Date,Description,Debit,Credit,Type\n"
        "2023-06-12,Debit col with CR type,450.00,,CR\n"
        "2023-06-13,Credit col with DR type,,950.00,DR\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "contra_col_type.csv", "doc-c-col-type")
    assert len(txns) == 0
    assert len(evidence) == 2
    for ev in evidence:
        assert ev.amount is None
        assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in ev.warnings)
        assert any("contradictory direction" in w.message.lower() for w in ev.warnings)


def test_csv_ambiguous_positive_amount_without_direction_rejected():
    """
    Issue 3.7: An ambiguous positive amount with no reliable direction indicator
    must NOT default to credit; it must be rejected with a structured warning.
    """
    csv_data = (
        "Date,Description,Amount\n"
        "2023-06-14,Ambiguous transaction ref 12345,750.00\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "ambig_amt.csv", "doc-ambig-amt")
    assert len(txns) == 0
    assert len(evidence) == 1
    assert evidence[0].amount is None
    assert any(w.code == WarningCodes.AMBIGUOUS_COLUMN for w in evidence[0].warnings)
    assert any("ambiguous transaction direction" in w.message.lower() for w in evidence[0].warnings)


def test_csv_valid_debit_and_credit_rows():
    """
    Issue 3.8: Valid debit and credit rows parse correctly, canonical amount is
    strictly positive, and direction is represented solely through TransactionType.
    """
    csv_data = (
        "Date,Description,Debit,Credit,Amount,Type\n"
        "2023-07-01,Separate debit col,100.00,,,\n"
        "2023-07-02,Separate credit col,,200.00,,\n"
        "2023-07-03,Amount with DR col,,,300.00,DR\n"
        "2023-07-04,Amount with CR col,,,400.00,CR\n"
        "2023-07-05,Negative signed debit,,,-500.00,\n"
        "2023-07-06,Narration debit keyword (Paid to merchant),,,600.00,\n"
        "2023-07-07,Narration credit keyword (Salary credited),,,700.00,\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "valid_dirs.csv", "doc-vdirs")
    assert len(txns) == 7
    assert len(evidence) == 7

    # All amounts must be strictly positive
    for t in txns:
        assert t.amount > 0

    assert txns[0].type == TransactionType.DEBIT and txns[0].amount == Decimal("100.00")
    assert txns[1].type == TransactionType.CREDIT and txns[1].amount == Decimal("200.00")
    assert txns[2].type == TransactionType.DEBIT and txns[2].amount == Decimal("300.00")
    assert txns[3].type == TransactionType.CREDIT and txns[3].amount == Decimal("400.00")
    assert txns[4].type == TransactionType.DEBIT and txns[4].amount == Decimal("500.00")
    assert txns[5].type == TransactionType.DEBIT and txns[5].amount == Decimal("600.00")
    assert txns[6].type == TransactionType.CREDIT and txns[6].amount == Decimal("700.00")


def test_csv_negative_balance_with_valid_transaction():
    """
    Issue 3.9: Preserve valid negative balance (overdraft). Negative balance
    is NOT itself a transaction direction contradiction.
    """
    csv_data = (
        "Date,Description,Debit,Credit,Balance\n"
        "2023-08-01,Overdraft interest charge,25.50,,-2500.75\n"
        "2023-08-02,Deposit reducing overdraft,,1000.00,-1500.75\n"
    ).encode("utf-8")

    txns, evidence = parse_csv_transactions(csv_data, "overdraft.csv", "doc-overdraft")
    assert len(txns) == 2
    assert len(evidence) == 2

    # Row 1: Debit with negative balance
    assert txns[0].amount == Decimal("25.50")
    assert txns[0].type == TransactionType.DEBIT
    assert txns[0].balance == Decimal("-2500.75")

    # Row 2: Credit with negative balance
    assert txns[1].amount == Decimal("1000.00")
    assert txns[1].type == TransactionType.CREDIT
    assert txns[1].balance == Decimal("-1500.75")


def test_every_rejected_row_emits_structured_warning_and_creates_no_bank_transaction():
    """
    Issue 3.10: For every rejected row across all error conditions, verify that
    it emits a structured warning item and does NOT produce any BankTransaction.
    """
    rejected_cases = [
        ("Date,Description,Debit,Credit\n2023-01-01,Both populated,100.00,200.00\n", WarningCodes.AMBIGUOUS_COLUMN),
        ("Date,Description,Debit\n2023-01-01,Negative debit,-50.00\n", WarningCodes.AMBIGUOUS_COLUMN),
        ("Date,Description,Credit\n2023-01-01,Negative credit,-50.00\n", WarningCodes.AMBIGUOUS_COLUMN),
        ("Date,Description,Amount,Type\n2023-01-01,Neg with CR,-100.00,CR\n", WarningCodes.AMBIGUOUS_COLUMN),
        ("Date,Description,Amount,Type\n2023-01-01,Conflicting types,100.00,DR/CR\n", WarningCodes.AMBIGUOUS_COLUMN),
        ("Date,Description,Debit,Type\n2023-01-01,Contradictory debit CR,100.00,CR\n", WarningCodes.AMBIGUOUS_COLUMN),
        ("Date,Description,Amount\n2023-01-01,Ambiguous,500.00\n", WarningCodes.AMBIGUOUS_COLUMN),
        ("Date,Description,Debit\n2023-01-01,Zero debit,0.00\n", WarningCodes.VALUE_NOT_FOUND),
        ("Date,Description,Amount\n2023-01-01,Malformed,invalid_amt\n", WarningCodes.VALUE_NOT_FOUND),
        ("Date,Description,Amount\n2023-01-01,Blank,\n", WarningCodes.VALUE_NOT_FOUND),
    ]

    for csv_str, expected_code in rejected_cases:
        txns, evidence = parse_csv_transactions(csv_str.encode("utf-8"), "test.csv", "doc-rej")
        assert len(txns) == 0, f"Expected 0 BankTransactions for:\n{csv_str}"
        assert len(evidence) == 1
        assert any(w.code == expected_code for w in evidence[0].warnings), (
            f"Expected warning code {expected_code} in evidence for:\n{csv_str}"
        )


# ==============================================================================
# PDF INGESTION TESTS (Member 1 Implementation)
# ==============================================================================
import fitz


def create_simple_pdf(pages_text: list) -> bytes:
    """Helper to create an in-memory PDF with specified text per page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_multipage_text_extraction_and_page_provenance():
    """
    Verify text-based PDF extracts page-by-page, preserves Page 1, Page 2 provenance,
    generates deterministic evidence IDs, and populates ExtractionResult / EvidenceRecord correctly.
    """
    page1 = (
        "LOAN SANCTION LETTER\n"
        "Loan Amount: Rs. 500,000.00\n"
        "Net Disbursement: Rs. 490,000.00\n"
        "Interest Rate: 12.5%\n"
    )
    page2 = (
        "REPAYMENT SCHEDULE\n"
        "Disclosed APR: 13.8%\n"
        "EMI Amount: Rs. 16,750.00\n"
        "Tenure: 36 months\n"
        "First EMI Date: 2024-05-01\n"
        "Processing Fee: Rs. 10,000.00\n"
        "Loan ID: LN-9876543210\n"
    )
    pdf_bytes = create_simple_pdf([page1, page2])
    test_key = b"dummy-test-key-32-bytes-long-1234"

    result = extract_document(
        file_content=pdf_bytes,
        filename="sanction_letter.pdf",
        document_id="doc-pdf-001",
        hmac_key=test_key,
    )

    assert isinstance(result, ExtractionResult)
    assert result.file_type == "pdf"
    assert result.document_id == "doc-pdf-001"
    assert result.filename == "sanction_letter.pdf"
    assert len(result.evidence) >= 8

    # Verify Page 1 records
    page1_records = [e for e in result.evidence if e.structured_source_location and e.structured_source_location.page_number == 1]
    assert len(page1_records) == 3
    assert all(e.source_location == "Page 1" for e in page1_records)
    assert any(e.type == EvidenceType.LOAN_AMOUNT and e.amount == Decimal("500000.00") for e in page1_records)
    assert any(e.type == EvidenceType.NET_DISBURSEMENT and e.amount == Decimal("490000.00") for e in page1_records)
    assert any(e.type == EvidenceType.INTEREST_RATE for e in page1_records)

    # Verify Page 2 records
    page2_records = [e for e in result.evidence if e.structured_source_location and e.structured_source_location.page_number == 2]
    assert len(page2_records) >= 5
    assert all(e.source_location == "Page 2" for e in page2_records)
    assert any(e.type == EvidenceType.DISCLOSED_APR for e in page2_records)
    assert any(e.type == EvidenceType.EMI_AMOUNT and e.amount == Decimal("16750.00") for e in page2_records)
    assert any(e.type == EvidenceType.TENURE_MONTHS and e.value == "36" for e in page2_records)
    assert any(e.type == EvidenceType.FIRST_EMI_DATE for e in page2_records)
    assert any(e.type == EvidenceType.PROCESSING_FEE and e.amount == Decimal("10000.00") for e in page2_records)

    # Deterministic evidence ID verification: repeated call produces identical IDs
    result2 = extract_document(
        file_content=pdf_bytes,
        filename="sanction_letter.pdf",
        document_id="doc-pdf-001",
        hmac_key=test_key,
    )
    assert [e.id for e in result.evidence] == [e.id for e in result2.evidence]


def test_pdf_missing_values_stay_missing_no_fabrication():
    """
    Ensure missing financial terms in PDF remain missing/None.
    No fabricated zero amounts, no invented APRs, no defaults.
    """
    page1 = (
        "GENERAL INQUIRY NOTE\n"
        "Customer inquired about personal loan terms.\n"
        "Tenure: 24 months\n"
    )
    pdf_bytes = create_simple_pdf([page1])

    result = extract_document(
        file_content=pdf_bytes,
        filename="inquiry.pdf",
        document_id="doc-pdf-inq",
    )
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.type == EvidenceType.TENURE_MONTHS
    assert ev.value == "24"

    # Verify no loan amount, net disbursement, APR, EMI, or processing fee were fabricated
    types = [e.type for e in result.evidence]
    assert EvidenceType.LOAN_AMOUNT not in types
    assert EvidenceType.NET_DISBURSEMENT not in types
    assert EvidenceType.DISCLOSED_APR not in types
    assert EvidenceType.EMI_AMOUNT not in types
    assert EvidenceType.PROCESSING_FEE not in types


def test_pdf_encrypted_fails_explicitly():
    """Password-protected / encrypted PDF must raise EncryptedPDFError."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Confidential Sanction Letter")
    pdf_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner_secret",
        user_pw="user_secret",
    )
    doc.close()

    with pytest.raises(EncryptedPDFError):
        extract_document(
            file_content=pdf_bytes,
            filename="protected.pdf",
            document_id="doc-enc",
        )


def test_pdf_corrupted_file_fails_explicitly():
    """Malformed or invalid bytes with .pdf extension must raise CorruptedFileError or UnsupportedFileTypeError."""
    # Invalid magic bytes caught by validation
    with pytest.raises(UnsupportedFileTypeError):
        extract_document(b"Not a real pdf at all", "corrupted.pdf", "doc-corrupt")

    # Correct magic bytes %PDF- but corrupted content
    corrupted_bytes = b"%PDF-1.4\nGarbage data not conforming to valid PDF xref table %%EOF"
    with pytest.raises(CorruptedFileError):
        extract_document(corrupted_bytes, "broken.pdf", "doc-broken")


def test_pdf_empty_document_fails_explicitly():
    """Genuinely empty PDF (page with 0 text and 0 images) must raise EmptyDocumentError."""
    doc = fitz.open()
    doc.new_page()  # Blank page without any text or images
    pdf_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(EmptyDocumentError):
        extract_document(pdf_bytes, "blank.pdf", "doc-blank")


def test_pdf_scanned_image_only_fails_explicitly():
    """Scanned/image-only PDF with images and no extractable text must raise UnsupportedScannedPDFError."""
    doc = fitz.open()
    page = doc.new_page()
    # Insert a minimal synthetic image onto the page
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), False)
    page.insert_image(fitz.Rect(50, 50, 150, 150), pixmap=pix)
    pdf_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(UnsupportedScannedPDFError):
        extract_document(pdf_bytes, "scanned_doc.pdf", "doc-scanned")


def test_pdf_reference_fingerprinting_and_privacy(monkeypatch):
    """
    Test reference masking and HMAC fingerprinting from PDF documents:
    - References are masked to last 4 chars in reference_number / normalized_value.
    - Free text in source_excerpt / value is sanitized.
    - Fingerprints are deterministically computed.
    - Fails explicitly with MissingHMACKeyError when references are found without key.
    """
    sensitive_account = "987654321098"
    page_text = (
        f"SANCTION DETAILS\n"
        f"Loan Amount: Rs. 250,000.00\n"
        f"Account No: {sensitive_account}\n"
    )
    pdf_bytes = create_simple_pdf([page_text])
    test_key = b"hmac-key-for-privacy-verification"

    # With valid key:
    res = extract_document(pdf_bytes, "loan.pdf", "doc-priv", hmac_key=test_key)
    account_records = [e for e in res.evidence if e.type == EvidenceType.BANK_ACCOUNT_MENTION]
    assert len(account_records) == 1
    ev = account_records[0]

    # Masked: last 4 digits only
    assert ev.reference_number == "XXXX1098"
    assert sensitive_account not in ev.reference_number
    assert sensitive_account not in ev.source_excerpt
    assert ev.reference_fingerprint is not None

    # Without HMAC key when references exist: must fail closed with MissingHMACKeyError
    monkeypatch.delenv("PROOFLEDGER_REFERENCE_HMAC_KEY", raising=False)
    with pytest.raises(MissingHMACKeyError):
        extract_document(pdf_bytes, "loan.pdf", "doc-priv", hmac_key=None)


def test_pdf_no_references_does_not_require_hmac_key(monkeypatch):
    """PDF without reference numbers does not require HMAC key."""
    monkeypatch.delenv("PROOFLEDGER_REFERENCE_HMAC_KEY", raising=False)
    page_text = (
        "SANCTION DETAILS\n"
        "Loan Amount: Rs. 100,000.00\n"
        "Interest Rate: 10.0%\n"
    )
    pdf_bytes = create_simple_pdf([page_text])
    res = extract_document(pdf_bytes, "simple_loan.pdf", "doc-no-key", hmac_key=None)
    assert len(res.evidence) == 2


def test_member1_ingestion_end_to_end_integration():
    """
    Minimal end-to-end integration test for Member 1's ingestion service:
    1. Tests synthetic KFS-style PDF and bank transaction CSV through extract_document().
    2. Verifies ExtractionResult conforms to schema v0.2.0.
    3. Verifies source provenance (e.g. Page 1, row 1), stable deterministic IDs,
       and that missing financial values are not fabricated.
    4. Confirms privacy masking and HMAC fingerprinting behavior.
    """
    test_key = b"integration-test-secret-key-32b"

    # --- Step 1: Ingest Synthetic KFS PDF ---
    pdf_page_content = (
        "KEY FACT STATEMENT (KFS)\n"
        "Sanctioned Loan Amount: Rs. 350,000.00\n"
        "Net Disbursement: Rs. 343,000.00\n"
        "Annual Percentage Rate: 14.25%\n"
        "Loan Tenure: 24 months\n"
        "Monthly EMI: Rs. 16,840.00\n"
        "Loan Ref: LN-SYNTH-998877\n"
        # Processing fee is intentionally omitted in source to verify no fabrication
    )
    pdf_bytes = create_simple_pdf([pdf_page_content])
    pdf_result = extract_document(
        file_content=pdf_bytes,
        filename="synthetic_kfs.pdf",
        document_id="doc-synth-pdf",
        hmac_key=test_key,
    )

    # Conformance to schema v0.2.0
    assert isinstance(pdf_result, ExtractionResult)
    assert pdf_result.file_type == "pdf"
    assert pdf_result.document_id == "doc-synth-pdf"
    assert pdf_result.filename == "synthetic_kfs.pdf"

    # Verify extracted terms
    ev_types = {e.type for e in pdf_result.evidence}
    assert EvidenceType.LOAN_AMOUNT in ev_types
    assert EvidenceType.NET_DISBURSEMENT in ev_types
    assert EvidenceType.DISCLOSED_APR in ev_types
    assert EvidenceType.TENURE_MONTHS in ev_types
    assert EvidenceType.EMI_AMOUNT in ev_types
    assert EvidenceType.LOAN_MENTION in ev_types

    # Missing value not fabricated: processing fee was not in PDF, must not exist
    assert EvidenceType.PROCESSING_FEE not in ev_types

    # Provenance and stable deterministic IDs
    for ev in pdf_result.evidence:
        assert isinstance(ev.id, str)
        assert ev.id.startswith("ev_")
        assert ev.source_location == "Page 1"
        assert ev.structured_source_location is not None
        assert ev.structured_source_location.page_number == 1
        assert ev.verification_status == VerificationStatus.EXTRACTED

    # Privacy and HMAC on PDF reference
    loan_mention = next(e for e in pdf_result.evidence if e.type == EvidenceType.LOAN_MENTION)
    assert loan_mention.reference_number == "XXXX8877"
    assert "998877" not in (loan_mention.reference_number or "")
    assert loan_mention.reference_fingerprint is not None

    # Idempotency / stability of IDs
    pdf_result_2 = extract_document(
        file_content=pdf_bytes,
        filename="synthetic_kfs.pdf",
        document_id="doc-synth-pdf",
        hmac_key=test_key,
    )
    assert [e.id for e in pdf_result.evidence] == [e.id for e in pdf_result_2.evidence]

    # --- Step 2: Ingest Synthetic Bank Statement CSV ---
    csv_data = (
        "Date,Description,Debit,Credit,Balance,Ref No\n"
        "2024-01-10,Disbursement credit from lender,,343000.00,345000.00,REFDISB0123\n"
        "2024-02-10,Automated EMI debit,16840.00,,328160.00,REFEMI0456\n"
    ).encode("utf-8")

    csv_result = extract_document(
        file_content=csv_data,
        filename="synthetic_statement.csv",
        document_id="doc-synth-csv",
        hmac_key=test_key,
    )

    # Conformance to schema v0.2.0
    assert isinstance(csv_result, ExtractionResult)
    assert csv_result.file_type == "csv"
    assert len(csv_result.transactions) == 2
    assert len(csv_result.evidence) == 2

    # Verify transactions & direction
    disb_txn = csv_result.transactions[0]
    assert disb_txn.type == TransactionType.CREDIT
    assert disb_txn.amount == Decimal("343000.00")
    assert disb_txn.source_location == "row 1"
    assert disb_txn.reference_number == "XXXX0123"

    emi_txn = csv_result.transactions[1]
    assert emi_txn.type == TransactionType.DEBIT
    assert emi_txn.amount == Decimal("16840.00")
    assert emi_txn.source_location == "row 2"
    assert emi_txn.reference_number == "XXXX0456"

    # Evidence provenance and privacy
    for ev in csv_result.evidence:
        assert isinstance(ev.id, str)
        assert ev.id.startswith("ev_")
        assert ev.source_location in ("row 1", "row 2")
        assert ev.verification_status == VerificationStatus.EXTRACTED
        assert ev.reference_fingerprint is not None
        assert ev.amount is not None






