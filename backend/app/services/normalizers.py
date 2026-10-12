import re
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple
from backend.app.schemas import (
    ProcessingFee,
    ProcessingFeeModel,
    WarningItem,
    StructuredWarning,
)


def normalize_amount(raw_val: Optional[str]) -> Optional[Decimal]:
    """
    Deterministically normalizes a monetary string into a positive Decimal.
    Handles currency symbols ($, Rs, ₹, EUR, etc.), commas, negative parentheses.
    Does NOT convert missing or empty values into zero.
    """
    if raw_val is None:
        return None
    
    val = str(raw_val).strip()
    if not val:
        return None
    
    # Check for empty indicators or dashes
    if val in ("-", "--", "N/A", "NA", "null", "None", "."):
        return None
    
    # Remove currency symbols and formatting prefixes
    # ₹, Rs., Rs, $, €, £, etc.
    cleaned = re.sub(r'[₹$€£]', '', val)
    cleaned = re.sub(r'\b(Rs\.?|INR|USD|EUR|GBP)\b\.?', '', cleaned, flags=re.IGNORECASE).strip()
    # Strip any leading period or leftover currency punctuation
    cleaned = cleaned.lstrip('.').strip()
    
    # Check for negative format in parentheses: (100.00) or -100.00
    is_negative = False
    if cleaned.startswith('(') and cleaned.endswith(')'):
        is_negative = True
        cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith('-'):
        is_negative = True
        cleaned = cleaned[1:].strip()
    elif cleaned.endswith('-'):
        is_negative = True
        cleaned = cleaned[:-1].strip()
        
    # Remove all commas (supports standard and Indian format e.g. 4,50,000.00)
    cleaned = cleaned.replace(',', '').strip()
    
    # Extract numeric part
    match = re.search(r'^\d+(\.\d+)?$', cleaned)
    if not match:
        return None
        
    try:
        dec = Decimal(match.group(0))
        return dec
    except InvalidOperation:
        return None


def parse_signed_amount(raw_val: Optional[str]) -> Tuple[Optional[Decimal], Optional[bool], bool]:
    """
    Parses a monetary string while explicitly preserving sign information.
    Returns:
        (magnitude_decimal, is_negative_or_None, is_valid_bool)
    - If raw_val is missing, empty, or placeholder ('-', 'N/A'): returns (None, None, True)
    - If raw_val is malformed / unparseable: returns (None, None, False)
    - If raw_val parses successfully: returns (abs_decimal, is_negative, True)
    """
    if raw_val is None:
        return None, None, True

    val = str(raw_val).strip()
    if not val:
        return None, None, True

    if val in ("-", "--", "N/A", "NA", "null", "None", "."):
        return None, None, True

    # Strip currency symbols and codes
    cleaned = re.sub(r'[₹$€£]', '', val)
    cleaned = re.sub(r'\b(Rs\.?|INR|USD|EUR|GBP)\b\.?', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.lstrip('.').strip()

    is_negative = False
    if cleaned.startswith('(') and cleaned.endswith(')'):
        is_negative = True
        cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith('-'):
        is_negative = True
        cleaned = cleaned[1:].strip()
    elif cleaned.endswith('-'):
        is_negative = True
        cleaned = cleaned[:-1].strip()

    cleaned = cleaned.replace(',', '').strip()

    match = re.search(r'^\d+(\.\d+)?$', cleaned)
    if not match:
        return None, None, False

    try:
        dec = Decimal(match.group(0))
        return dec, is_negative, True
    except InvalidOperation:
        return None, None, False


def normalize_percentage(raw_val: Optional[str]) -> Optional[Decimal]:
    """
    Deterministically normalizes percentage string into a Decimal percentage rate.
    e.g., '12.5%' -> Decimal('12.5'), '2 %' -> Decimal('2').
    Returns None if missing or unparseable.
    """
    if raw_val is None:
        return None
    val = str(raw_val).strip()
    if not val:
        return None
    
    cleaned = val.replace('%', '').strip()
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def normalize_date(raw_date_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Deterministically parses a date string into an ISO-format string (YYYY-MM-DD).
    Strictly handles unambiguous dates.
    If ambiguous (e.g. 05/06/2023 where day could be 5 or 6) and format cannot be
    inferred unambiguously from separator or surrounding cues, does NOT guess:
    returns (None, "Ambiguous date format: cannot distinguish day from month").
    
    Returns tuple of (iso_date_string_or_None, warning_or_None).
    """
    if not raw_date_str:
        return None, None
    
    val = str(raw_date_str).strip()
    if not val or val in ("-", "--", "N/A", "NA"):
        return None, None

    # Try ISO format first (YYYY-MM-DD or YYYY/MM/DD)
    iso_match = re.match(r'^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$', val)
    if iso_match:
        y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
        try:
            return date(y, m, d).isoformat(), None
        except ValueError:
            return None, f"Invalid calendar date: {val}"

    # Try Month name patterns e.g., "15-Jan-2023", "Jan 15, 2023", "15 Jan 2023", "2023-Jan-15"
    for fmt in ("%d-%b-%Y", "%d/%b/%Y", "%d %b %Y", "%d-%B-%Y", "%d %B %Y", "%b %d, %Y", "%b-%d-%Y", "%Y-%b-%d"):
        try:
            dt = datetime.strptime(val, fmt)
            return dt.date().isoformat(), None
        except ValueError:
            pass

    # Delimited dates: DD/MM/YYYY or MM/DD/YYYY or DD-MM-YYYY
    match_delim = re.match(r'^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$', val)
    if match_delim:
        p1, p2, p3 = int(match_delim.group(1)), int(match_delim.group(2)), int(match_delim.group(3))
        
        # Normalize 2-digit year to 20XX
        year = p3 if p3 >= 100 else 2000 + p3
        
        # If one part is > 12, it MUST be the day
        if p1 > 12 and p2 <= 12:
            # DD/MM/YYYY
            try:
                return date(year, p2, p1).isoformat(), None
            except ValueError:
                return None, f"Invalid calendar date: {val}"
        elif p2 > 12 and p1 <= 12:
            # MM/DD/YYYY
            try:
                return date(year, p1, p2).isoformat(), None
            except ValueError:
                return None, f"Invalid calendar date: {val}"
        elif p1 <= 12 and p2 <= 12:
            # If day and month are identical (e.g. 05/05/2023), unambiguous!
            if p1 == p2:
                try:
                    return date(year, p1, p2).isoformat(), None
                except ValueError:
                    return None, f"Invalid calendar date: {val}"
            # Both <= 12 and different: AMBIGUOUS! Do NOT guess.
            return None, f"Ambiguous date '{val}': cannot determine whether {p1} or {p2} is day or month."

    return None, f"Unrecognized date format: {val}"


def parse_processing_fee(
    raw_text: Optional[str],
    evidence_ids: Optional[list[str]] = None
) -> Optional[ProcessingFee]:
    """
    Parses a processing fee string into a typed ProcessingFee (canonical schema).
    Distinguishes fixed amounts from percentage rates.
    Correctly recognizes GST treatments ('+ GST', 'inclusive of GST', etc.).
    Preserves original wording and never equates a percentage (e.g. '2% + GST') to a fixed amount.
    """
    if not raw_text:
        return None
    
    text = str(raw_text).strip()
    if not text:
        return None
    
    eids = list(evidence_ids or ["ev_fee_default"])
    warnings: list[StructuredWarning] = []
    gst_treatment = "unspecified"
    
    lower = text.lower()
    if "+ gst" in lower or "plus gst" in lower or "exclusive of gst" in lower or "+ applicable taxes" in lower:
        gst_treatment = "additional"
    elif "incl. of gst" in lower or "including gst" in lower or "inclusive of gst" in lower:
        gst_treatment = "included"

    # Check for percentage fee: e.g. "2%", "1.5 % + GST", "Up to 2% of loan amount"
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if pct_match:
        pct_val = Decimal(pct_match.group(1))
        # Look for basis: e.g. "of loan amount"
        basis = None
        basis_match = re.search(r'of\s+([a-zA-Z\s]+)', text, flags=re.IGNORECASE)
        if basis_match:
            basis = basis_match.group(1).strip()
            
        return ProcessingFee(
            fee_type="percentage",
            original_text=text,
            normalized_value=None,  # Not converted to flat amount without loan amount context!
            percentage_rate=pct_val,
            basis=basis,
            currency=None,
            gst_treatment=gst_treatment,
            evidence_ids=eids,
            warnings=warnings
        )

    # Check for fixed monetary fee: e.g. "Rs. 2500", "2500 + GST", "₹ 1,000", "Rs. 1,500 inclusive of GST"
    # First try direct normalize_amount
    amount = normalize_amount(text)
    if amount is None:
        # Extract numeric/amount chunk from the fee text (e.g. "1,500" from "Rs. 1,500 inclusive of GST")
        fixed_match = re.search(r'(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)', text, flags=re.IGNORECASE)
        if fixed_match:
            amount = normalize_amount(fixed_match.group(1))

    if amount is not None:
        # Currency identification
        currency = "INR" if ("rs" in lower or "₹" in lower or "inr" in lower) else None
        return ProcessingFee(
            fee_type="fixed",
            value=amount,
            original_text=text,
            normalized_value=amount,
            percentage_rate=None,
            basis=None,
            currency=currency,
            gst_treatment=gst_treatment,
            evidence_ids=eids,
            warnings=warnings
        )

    # If neither could be parsed deterministically, record warning
    warnings.append(StructuredWarning(
        code="FEE_AMOUNT_MISSING",
        message=f"Could not classify fee structure from text: '{text}'",
        evidence_ids=eids
    ))
    return ProcessingFee(
        fee_type="fixed",
        value=None,
        original_text=text,
        normalized_value=None,
        percentage_rate=None,
        basis=None,
        currency=None,
        gst_treatment=gst_treatment,
        evidence_ids=eids,
        warnings=warnings
    )

