import pytest
from decimal import Decimal

from backend.app.services.normalizers import (
    normalize_amount,
    normalize_percentage,
    normalize_date,
    parse_processing_fee,
)


def test_normalize_amount_various_formats():
    assert normalize_amount("1,500.50") == Decimal("1500.50")
    assert normalize_amount("₹ 25,000") == Decimal("25000")
    assert normalize_amount("Rs. 4,50,000.00") == Decimal("450000.00")
    assert normalize_amount("$100.99") == Decimal("100.99")
    assert normalize_amount("(500.00)") == Decimal("500.00")  # Returns positive amount
    assert normalize_amount("-250.75") == Decimal("250.75")


def test_normalize_amount_missing_and_malformed():
    """Missing or empty values must return None, not Decimal('0')."""
    assert normalize_amount(None) is None
    assert normalize_amount("") is None
    assert normalize_amount("   ") is None
    assert normalize_amount("-") is None
    assert normalize_amount("N/A") is None
    assert normalize_amount("invalid_number") is None


def test_normalize_percentage():
    assert normalize_percentage("12.5%") == Decimal("12.5")
    assert normalize_percentage(" 8.75 % ") == Decimal("8.75")
    assert normalize_percentage(None) is None
    assert normalize_percentage("NA") is None


def test_normalize_date_unambiguous():
    # ISO formats
    d, warn = normalize_date("2023-08-25")
    assert d == "2023-08-25"
    assert warn is None

    # Month name
    d, warn = normalize_date("15-Jan-2023")
    assert d == "2023-01-15"
    assert warn is None

    # Clear DD/MM/YYYY where day > 12
    d, warn = normalize_date("25/04/2023")
    assert d == "2023-04-25"
    assert warn is None

    # Clear MM/DD/YYYY where day > 12
    d, warn = normalize_date("04/25/2023")
    assert d == "2023-04-25"
    assert warn is None

    # Identical day and month
    d, warn = normalize_date("07/07/2023")
    assert d == "2023-07-07"
    assert warn is None


def test_normalize_date_ambiguous_does_not_guess():
    """Ambiguous dates like 05/06/2023 must return None and a warning."""
    d, warn = normalize_date("05/06/2023")
    assert d is None
    assert warn is not None
    assert "Ambiguous date" in warn


def test_parse_processing_fee_percentage_with_gst():
    fee = parse_processing_fee("2% + GST of loan amount", evidence_ids=["ev_1"])
    assert fee is not None
    assert fee.fee_type == "percentage"
    assert fee.percentage_rate == Decimal("2")
    assert fee.normalized_value is None  # Never converted to 2 flat units
    assert fee.gst_treatment == "additional"
    assert fee.basis == "loan amount"
    assert fee.evidence_ids == ["ev_1"]


def test_parse_processing_fee_fixed_amount():
    fee = parse_processing_fee("Rs. 1,500 inclusive of GST", evidence_ids=["ev_2"])
    assert fee is not None
    assert fee.fee_type == "fixed"
    assert fee.normalized_value == Decimal("1500")
    assert fee.percentage_rate is None
    assert fee.currency == "INR"
    assert fee.gst_treatment == "included"
    assert fee.evidence_ids == ["ev_2"]


def test_parse_processing_fee_unparseable():
    fee = parse_processing_fee("Subject to terms and conditions", evidence_ids=["ev_3"])
    assert fee is not None
    assert fee.normalized_value is None
    assert len(fee.warnings) > 0
    assert fee.warnings[0].code == "FEE_AMOUNT_MISSING"

