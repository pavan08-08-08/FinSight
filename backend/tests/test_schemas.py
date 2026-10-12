import datetime as dt
from decimal import Decimal
import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    EvidenceRecord,
    SourceLocation,
    WarningItem,
    StructuredWarning,
    ProcessingFee,
    ProcessingFeeModel,
    BankTransaction,
    NormalizedTransaction,
    IngestionValidationResult,
    ExtractionResult,
    VerificationStatus,
    EvidenceType,
    TransactionType,
    FeeType,
    make_evidence_id,
)


def test_evidence_record_valid_core_and_optional_fields():
    """Verify that EvidenceRecord initializes with both core and optional Afterlife fields."""
    loc = SourceLocation(page_number=1, row_index=5, raw_source_snippet="Loan Amount: Rs. 50,000")
    warning = WarningItem(code="TEST_WARN", message="Warning description", evidence_ids=["ev_1"])
    
    rec = EvidenceRecord(
        id="ev_loan_12345",
        document_id="doc_uuid_999",
        type="loan_amount",
        value="50000.00",
        source_document="kfs_sample.txt",
        source_location=loc.to_canonical_string(),
        confidence=0.95,
        verification_status="extracted",
        label="Sanctioned Loan Amount",
        normalized_value="50000.00",
        extraction_method="deterministic",
        confidence_basis="Regex matched KFS header",
        warnings=[warning],
        supporting_evidence_ids=["ev_supp_01"],
        # Optional Afterlife fields
        category="bank_account",
        institution="HDFC Bank",
        reference_number="XXXX4321",
        reference_fingerprint="a" * 64,
        amount=Decimal("50000.00"),
        currency="INR",
        date=dt.date(2023, 5, 15),
        maturity_date=dt.date(2026, 5, 15),
        stated_status="active",
        source_excerpt="Excerpt of document line"
    )

    assert rec.id == "ev_loan_12345"
    assert rec.confidence == 0.95
    assert rec.verification_status == VerificationStatus.EXTRACTED
    assert rec.amount == Decimal("50000.00")
    assert len(rec.warnings) == 1
    assert rec.warnings[0].code == "TEST_WARN"


def test_evidence_record_confidence_is_nullable():
    """Confidence must be nullable when confidence cannot be reliably assessed."""
    rec = EvidenceRecord(
        id="ev_001",
        document_id="doc_001",
        type="financial_mention",
        value="Mentioned Account",
        source_document="doc.csv",
        source_location="row 0",
        confidence=None,  # Nullable
        verification_status="unverified"
    )
    assert rec.confidence is None


def test_verification_status_exact_canonical_values():
    """Only approved verification statuses must be accepted."""
    for valid_status in ["extracted", "needs_verification", "verified", "conflicting", "unverified"]:
        rec = EvidenceRecord(
            id="ev_status",
            document_id="doc_1",
            type="loan_amount",
            value="100",
            source_document="doc.txt",
            source_location="line 1",
            verification_status=valid_status
        )
        assert rec.verification_status == valid_status

    with pytest.raises(ValidationError):
        EvidenceRecord(
            id="ev_status",
            document_id="doc_1",
            type="loan_amount",
            value="100",
            source_document="doc.txt",
            source_location="line 1",
            verification_status="invalid_custom_status"
        )


def test_processing_fee_model_percentage_preserves_wording():
    """Processing fee model preserves original percentage wording and does not fabricate flat amount."""
    fee = ProcessingFee(
        fee_type=FeeType.PERCENTAGE,
        original_text="2% + GST of sanctioned amount",
        normalized_value=None,  # Must remain None without loan amount input
        percentage_rate=Decimal("2.0"),
        basis="sanctioned amount",
        currency=None,
        gst_treatment="additional",
        evidence_ids=["ev_fee_01"],
        warnings=[]
    )
    assert fee.fee_type is FeeType.PERCENTAGE
    assert fee.percentage_rate == Decimal("2.0")
    assert fee.normalized_value is None
    assert fee.gst_treatment == "additional"
    assert "2% + GST" in fee.original_text


def test_normalized_transaction_model():
    """BankTransaction / NormalizedTransaction requires positive Decimal, valid types, and optional fields."""
    txn = BankTransaction(
        id="ev_txn_123",
        date=dt.date(2023, 1, 15),
        description="Salary Deposit",
        amount=Decimal("50000.00"),
        type="credit",
        balance=Decimal("125000.50"),
        reference_number="XXXX1234",
        source_document="statement.csv",
        source_location="row 2"
    )
    assert txn.amount == Decimal("50000.00")
    assert txn.type == TransactionType.CREDIT
    assert txn.source_location == "row 2"

    with pytest.raises(ValidationError):
        BankTransaction(
            id="ev_txn_bad",
            date=dt.date(2023, 1, 15),
            description="Bad",
            amount=Decimal("500"),
            type="invalid_type",  # Only 'credit' or 'debit'
            source_document="statement.csv"
        )


def test_extraction_result_defaults_and_serialization():
    """Verify ExtractionResult default collections, non-statement empty transactions, and serialization."""
    res = ExtractionResult(
        document_id="doc_kfs_001",
        filename="kfs_doc.txt",
        file_type="txt"
    )
    assert res.document_id == "doc_kfs_001"
    assert res.filename == "kfs_doc.txt"
    assert res.file_type == "txt"
    assert res.evidence == []
    assert res.transactions == []  # Non-statement documents cleanly return empty transactions
    assert res.warnings == []
    assert res.metadata == {}

    # Verify JSON serialization round-trip
    dumped = res.model_dump()
    assert dumped["evidence"] == []
    assert dumped["transactions"] == []
    assert dumped["metadata"] == {}


def test_evidence_record_flexible_source_location():
    """Verify EvidenceRecord adapts SourceLocation objects and accepts strings."""
    # 1. Structured SourceLocation adapted to string
    loc_struct = SourceLocation(page_number=3, row_index=12, line_number=45, char_start=10, char_end=35)
    rec1 = EvidenceRecord(
        id="ev_01",
        document_id="doc_01",
        type="loan_amount",
        value="50000",
        source_document="doc.pdf",
        source_location=loc_struct
    )
    assert rec1.source_location == "Page 3, row 12, line 45, chars 10-35"

    # Convert back via adapter
    adapted_back = SourceLocation.from_canonical_string(rec1.source_location)
    assert adapted_back.page_number == 3
    assert adapted_back.row_index == 12

    # 2. String source_location (Member 3 Afterlife conventions)
    rec2 = EvidenceRecord(
        id="ev_02",
        document_id="doc_01",
        type="insurance_policy_mention",
        value="LIC Policy 998877",
        source_document="email.txt",
        source_location="Message 14"
    )
    assert rec2.source_location == "Message 14"

    rec3 = EvidenceRecord(
        id="ev_03",
        document_id="doc_01",
        type="bank_account_mention",
        value="SB Account",
        source_document="letter.pdf",
        source_location="Page 3"
    )
    assert rec3.source_location == "Page 3"


def test_evidence_record_preserves_internal_structured_provenance():
    """Verify that structured SourceLocation survives internally and serializes as canonical string."""
    loc = SourceLocation(
        page_number=2,
        line_number=10,
        raw_source_snippet="Sanctioned Amount: INR 10,000",
        char_start=0,
        char_end=30,
        description="Page 2, line 10"
    )
    rec = EvidenceRecord(
        id="ev_prov_1",
        document_id="doc_prov_1",
        type="loan_amount",
        value="10000",
        source_document="doc.pdf",
        source_location=loc
    )

    # Canonical string at public boundary
    assert rec.source_location == "Page 2, line 10"

    # Internal structured provenance available
    assert rec.structured_source_location is not None
    assert rec.structured_source_location.page_number == 2
    assert rec.structured_source_location.line_number == 10
    assert rec.structured_source_location.raw_source_snippet == "Sanctioned Amount: INR 10,000"
    assert rec.structured_source_location.char_start == 0
    assert rec.structured_source_location.char_end == 30

    # Serialization does not leak internal private fields
    dump = rec.model_dump()
    assert dump["source_location"] == "Page 2, line 10"
    assert "_source_location_obj" not in dump
    assert "structured_source_location" not in dump



