import io
import os
import re
import csv
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime as dt_datetime, date as dt_date

import fitz

from backend.app.schemas import (
    IngestionValidationResult,
    ExtractionResult,
    BankTransaction,
    NormalizedTransaction,
    EvidenceRecord,
    EvidenceType,
    VerificationStatus,
    TransactionType,
    SourceLocation,
    WarningItem,
    WarningCodes,
    normalize_reference,
)
from backend.app.services.errors import (
    FileSizeLimitExceededError,
    UnsupportedFileTypeError,
    CorruptedFileError,
    MissingHMACKeyError,
    EncryptedPDFError,
    EmptyDocumentError,
    UnsupportedScannedPDFError,
)
from backend.app.services.normalizers import (
    normalize_amount,
    parse_signed_amount,
    normalize_date,
    normalize_percentage,
    parse_processing_fee,
)
from backend.app.services.evidence_id import (
    mask_reference,
    sanitize_free_text,
    compute_reference_fingerprint,
    make_evidence_id,
)

# 10 MB per-file upload limit
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}


def validate_file_upload(file_content: bytes, filename: str) -> IngestionValidationResult:
    """
    Validates file extension, magic bytes, and size limit (<= 10MB).
    """
    if len(file_content) > MAX_FILE_SIZE_BYTES:
        raise FileSizeLimitExceededError(
            f"File '{filename}' exceeds the maximum allowed size of 10MB (got {len(file_content)} bytes)."
        )

    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{ext}'. ProofLedger supports: PDF, TXT, CSV."
        )

    # Magic byte checks
    if ext == ".pdf":
        if not file_content.startswith(b"%PDF-"):
            raise UnsupportedFileTypeError(f"File '{filename}' has .pdf extension but lacks '%PDF-' magic bytes.")
        file_type = "pdf"
    elif ext == ".csv":
        file_type = "csv"
    else:
        file_type = "txt"

    return IngestionValidationResult(
        filename=filename,
        file_type=file_type,
        file_size_bytes=len(file_content),
        is_valid=True,
    )


# Standard bank column header aliases
DATE_HEADERS = {"date", "txn date", "txndate", "transaction date", "value date", "post date", "trans date"}
DESC_HEADERS = {"description", "particulars", "narration", "transaction remarks", "details", "remarks"}
DEBIT_HEADERS = {"debit", "debit amount", "dr", "withdrawal", "withdrawals", "debit (inr)", "withdrawal amount"}
CREDIT_HEADERS = {"credit", "credit amount", "cr", "deposit", "deposits", "credit (inr)", "deposit amount"}
AMOUNT_HEADERS = {"amount", "txn amount", "transaction amount", "amount (inr)"}
TYPE_HEADERS = {"type", "txn type", "cr/dr", "d/c", "transaction type"}
BALANCE_HEADERS = {"balance", "closing balance", "running balance", "bal", "balance (inr)", "net balance"}
REF_HEADERS = {"ref no", "reference no", "ref", "chq / ref no", "cheque no", "reference number", "utr", "txn id", "transaction id"}


def _match_column(headers: List[str], target_set: set) -> Optional[int]:
    """Finds the index of a matching column name in a normalized header list."""
    for idx, h in enumerate(headers):
        clean_h = re.sub(r'[^a-zA-Z0-9/ ]', '', h.lower().strip())
        if clean_h in target_set:
            return idx
    # Partial word-boundary matching fallback (avoids 'cr' matching 'description')
    for idx, h in enumerate(headers):
        clean_h = re.sub(r'[^a-zA-Z0-9/ ]', '', h.lower().strip())
        for target in target_set:
            if re.search(r'\b' + re.escape(target) + r'\b', clean_h):
                return idx
    return None


def parse_csv_transactions(
    file_content: bytes,
    filename: str,
    document_id: str,
    hmac_key: Optional[bytes] = None
) -> Tuple[List[NormalizedTransaction], List[EvidenceRecord]]:
    """
    Parses bank transactions from a CSV document preserving raw cells internally.
    Normalizes each transaction to:
    - id
    - date (ISO or null if missing/ambiguous)
    - description
    - amount (positive normalized monetary string)
    - type ('credit' or 'debit')
    - balance (optional positive normalized monetary string)
    - reference_number (masked)
    - source_row_index provenance

    Returns:
    - List of NormalizedTransaction objects (for Member 2 / matching)
    - List of EvidenceRecord objects with canonical type 'repayment_transaction' or 'financial_mention'
    """
    try:
        # Decode UTF-8, with fallback to latin-1 for tricky bank statements
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            text = file_content.decode("latin-1")
    except Exception as e:
        raise CorruptedFileError(f"Could not decode CSV text from '{filename}': {str(e)}")

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        return [], []

    # Find the header row (skip preamble lines such as bank address / metadata)
    header_row_idx = -1
    date_col = desc_col = debit_col = credit_col = amount_col = type_col = bal_col = ref_col = None

    for r_idx, row in enumerate(rows):
        if not any(row):
            continue
        cleaned_cells = [c.strip() for c in row]
        # Check if this row looks like a transaction table header
        d_idx = _match_column(cleaned_cells, DATE_HEADERS)
        des_idx = _match_column(cleaned_cells, DESC_HEADERS)
        
        if d_idx is not None and des_idx is not None:
            header_row_idx = r_idx
            date_col = d_idx
            desc_col = des_idx
            debit_col = _match_column(cleaned_cells, DEBIT_HEADERS)
            credit_col = _match_column(cleaned_cells, CREDIT_HEADERS)
            amount_col = _match_column(cleaned_cells, AMOUNT_HEADERS)
            type_col = _match_column(cleaned_cells, TYPE_HEADERS)
            bal_col = _match_column(cleaned_cells, BALANCE_HEADERS)
            ref_col = _match_column(cleaned_cells, REF_HEADERS)
            break

    if header_row_idx == -1:
        # If no classic header found, treat row 0 as header
        header_row_idx = 0
        cleaned_cells = [c.strip() for c in rows[0]]
        date_col = _match_column(cleaned_cells, DATE_HEADERS)
        desc_col = _match_column(cleaned_cells, DESC_HEADERS)
        debit_col = _match_column(cleaned_cells, DEBIT_HEADERS)
        credit_col = _match_column(cleaned_cells, CREDIT_HEADERS)
        amount_col = _match_column(cleaned_cells, AMOUNT_HEADERS)
        type_col = _match_column(cleaned_cells, TYPE_HEADERS)
        bal_col = _match_column(cleaned_cells, BALANCE_HEADERS)
        ref_col = _match_column(cleaned_cells, REF_HEADERS)

    normalized_transactions: List[NormalizedTransaction] = []
    evidence_records: List[EvidenceRecord] = []

    for r_idx in range(header_row_idx + 1, len(rows)):
        raw_row = rows[r_idx]
        if not raw_row or not any(cell.strip() for cell in raw_row):
            continue  # Skip blank row

        def get_cell(col_idx: Optional[int]) -> Optional[str]:
            if col_idx is not None and 0 <= col_idx < len(raw_row):
                val = raw_row[col_idx].strip()
                return val if val else None
            return None

        raw_date = get_cell(date_col)
        raw_desc = get_cell(desc_col) or "UNKNOWN TRANSACTION"
        raw_debit = get_cell(debit_col)
        raw_credit = get_cell(credit_col)
        raw_amount = get_cell(amount_col)
        raw_type = get_cell(type_col)
        raw_bal = get_cell(bal_col)
        raw_ref = get_cell(ref_col)

        source_loc_str = f"row {r_idx}"
        ev_id = make_evidence_id(document_id, source_loc_str, 0)
        row_warnings: List[WarningItem] = []

        # Sanitize free-text description for privacy (conservative redaction of embedded account/phone numbers)
        sanitized_desc = sanitize_free_text(raw_desc) or "UNKNOWN TRANSACTION"

        # Determine transaction type and amount
        txn_type: Optional[str] = None
        txn_amount_dec: Optional[Decimal] = None

        has_separate_cols = (debit_col is not None or credit_col is not None)
        has_amount_col = (amount_col is not None)

        if not has_separate_cols and not has_amount_col:
            row_warnings.append(WarningItem(
                code=WarningCodes.VALUE_NOT_FOUND,
                message="Missing amount column: no debit, credit, or amount columns found in CSV.",
                evidence_ids=[ev_id]
            ))
        else:
            col_type: Optional[str] = None
            col_amt_dec: Optional[Decimal] = None
            has_col_contradiction = False

            if has_separate_cols:
                d_amt, d_neg, d_valid = parse_signed_amount(raw_debit) if debit_col is not None else (None, None, True)
                c_amt, c_neg, c_valid = parse_signed_amount(raw_credit) if credit_col is not None else (None, None, True)

                if not d_valid or not c_valid:
                    has_col_contradiction = True
                    row_warnings.append(WarningItem(
                        code=WarningCodes.VALUE_NOT_FOUND,
                        message="Malformed monetary value in debit or credit column.",
                        evidence_ids=[ev_id]
                    ))
                elif d_amt is not None and d_amt > 0 and d_neg is True:
                    has_col_contradiction = True
                    row_warnings.append(WarningItem(
                        code=WarningCodes.AMBIGUOUS_COLUMN,
                        message="Contradictory or ambiguous negative value in debit column.",
                        evidence_ids=[ev_id]
                    ))
                elif c_amt is not None and c_amt > 0 and c_neg is True:
                    has_col_contradiction = True
                    row_warnings.append(WarningItem(
                        code=WarningCodes.AMBIGUOUS_COLUMN,
                        message="Contradictory or ambiguous negative value in credit column.",
                        evidence_ids=[ev_id]
                    ))
                elif d_amt is not None and c_amt is not None and d_amt > 0 and c_amt > 0:
                    has_col_contradiction = True
                    row_warnings.append(WarningItem(
                        code=WarningCodes.AMBIGUOUS_COLUMN,
                        message="Contradictory values: both debit and credit columns contain non-zero amounts.",
                        evidence_ids=[ev_id]
                    ))
                elif (d_amt is not None and d_amt == 0) and (c_amt is not None and c_amt > 0):
                    col_type = "credit"
                    col_amt_dec = c_amt
                elif (c_amt is not None and c_amt == 0) and (d_amt is not None and d_amt > 0):
                    col_type = "debit"
                    col_amt_dec = d_amt
                elif (d_amt is not None and d_amt == 0) or (c_amt is not None and c_amt == 0):
                    has_col_contradiction = True
                    row_warnings.append(WarningItem(
                        code=WarningCodes.VALUE_NOT_FOUND,
                        message="Transaction amount is zero; zero-amount rows cannot form a valid BankTransaction.",
                        evidence_ids=[ev_id]
                    ))
                elif d_amt is not None and d_amt > 0:
                    col_type = "debit"
                    col_amt_dec = d_amt
                elif c_amt is not None and c_amt > 0:
                    col_type = "credit"
                    col_amt_dec = c_amt

            # Check explicit type column
            explicit_type: Optional[str] = None
            if raw_type:
                clean_type = raw_type.strip().lower()
                has_dr_indicator = bool(re.search(r'\b(dr|debit|d|withdrawal|payment)\b', clean_type))
                has_cr_indicator = bool(re.search(r'\b(cr|credit|c|deposit|receipt)\b', clean_type))
                if has_dr_indicator and has_cr_indicator:
                    has_col_contradiction = True
                    row_warnings.append(WarningItem(
                        code=WarningCodes.AMBIGUOUS_COLUMN,
                        message="Conflicting explicit type indicators: contains both debit and credit indicators.",
                        evidence_ids=[ev_id]
                    ))
                elif has_dr_indicator:
                    explicit_type = "debit"
                elif has_cr_indicator:
                    explicit_type = "credit"
                else:
                    has_col_contradiction = True
                    row_warnings.append(WarningItem(
                        code=WarningCodes.AMBIGUOUS_COLUMN,
                        message=f"Unrecognized transaction type indicator: '{clean_type}'.",
                        evidence_ids=[ev_id]
                    ))

            # Contradiction between separate column direction and explicit type indicator
            if col_type is not None and explicit_type is not None and col_type != explicit_type:
                has_col_contradiction = True
                row_warnings.append(WarningItem(
                    code=WarningCodes.AMBIGUOUS_COLUMN,
                    message=f"Contradictory direction: {col_type} column populated but type indicator specifies {explicit_type}.",
                    evidence_ids=[ev_id]
                ))

            # Separate amount column parsing
            amt_val: Optional[Decimal] = None
            amt_neg: Optional[bool] = None
            if has_amount_col:
                amt_val, amt_neg, amt_valid = parse_signed_amount(raw_amount)
                if not amt_valid:
                    has_col_contradiction = True
                    row_warnings.append(WarningItem(
                        code=WarningCodes.VALUE_NOT_FOUND,
                        message="Malformed monetary value in amount column.",
                        evidence_ids=[ev_id]
                    ))
                elif amt_val is not None:
                    if amt_val == 0:
                        has_col_contradiction = True
                        row_warnings.append(WarningItem(
                            code=WarningCodes.VALUE_NOT_FOUND,
                            message="Transaction amount is zero; zero-amount rows cannot form a valid BankTransaction.",
                            evidence_ids=[ev_id]
                        ))
                    elif amt_neg is True and explicit_type == "credit":
                        has_col_contradiction = True
                        row_warnings.append(WarningItem(
                            code=WarningCodes.AMBIGUOUS_COLUMN,
                            message="Contradictory direction: negative amount combined with credit type indicator.",
                            evidence_ids=[ev_id]
                        ))
                    elif amt_neg is True and col_type == "credit":
                        has_col_contradiction = True
                        row_warnings.append(WarningItem(
                            code=WarningCodes.AMBIGUOUS_COLUMN,
                            message="Contradictory direction: negative amount combined with credit column.",
                            evidence_ids=[ev_id]
                        ))
                    elif col_amt_dec is not None and amt_val != col_amt_dec:
                        has_col_contradiction = True
                        row_warnings.append(WarningItem(
                            code=WarningCodes.AMBIGUOUS_COLUMN,
                            message="Contradictory amounts: separate column amount does not match amount column value.",
                            evidence_ids=[ev_id]
                        ))

            # Check for completely missing amounts across all columns
            if not row_warnings and not has_col_contradiction:
                if not has_separate_cols and amt_val is None:
                    row_warnings.append(WarningItem(
                        code=WarningCodes.VALUE_NOT_FOUND,
                        message="Missing amount in transaction row.",
                        evidence_ids=[ev_id]
                    ))
                elif has_separate_cols and not has_amount_col and col_amt_dec is None:
                    row_warnings.append(WarningItem(
                        code=WarningCodes.VALUE_NOT_FOUND,
                        message="No transaction amount found in debit or credit column.",
                        evidence_ids=[ev_id]
                    ))
                elif has_separate_cols and has_amount_col and col_amt_dec is None and amt_val is None:
                    row_warnings.append(WarningItem(
                        code=WarningCodes.VALUE_NOT_FOUND,
                        message="Missing amount: debit, credit, and amount cells are blank.",
                        evidence_ids=[ev_id]
                    ))

            # If no warnings and no contradictions, determine final amount and type
            if not row_warnings and not has_col_contradiction:
                if col_amt_dec is not None:
                    txn_amount_dec = col_amt_dec
                    txn_type = col_type
                elif amt_val is not None and amt_val > 0:
                    txn_amount_dec = amt_val
                    if explicit_type is not None:
                        txn_type = explicit_type
                    elif amt_neg is True:
                        txn_type = "debit"
                    else:
                        clean_narration = (raw_desc or "").lower()
                        if re.search(r'\b(?:transfer\s+to|paid\s+to|payment\s+to|payment\s+for|debit|withdrawn|withdrawal|purchase|fee|charges?)\b', clean_narration):
                            txn_type = "debit"
                        elif re.search(r'\b(?:received\s+from|transfer\s+from|credited|salary|refund|deposit)\b', clean_narration):
                            txn_type = "credit"
                        else:
                            row_warnings.append(WarningItem(
                                code=WarningCodes.AMBIGUOUS_COLUMN,
                                message="Ambiguous transaction direction: positive amount with no explicit credit/debit indicator.",
                                evidence_ids=[ev_id]
                            ))
                            txn_amount_dec = None
                            txn_type = None

        # Balance normalization (supports negative balance for overdrafts)
        norm_bal_dec: Optional[Decimal] = None
        if bal_col is not None and raw_bal:
            b_val, b_neg, b_valid = parse_signed_amount(raw_bal)
            if b_valid and b_val is not None:
                norm_bal_dec = -b_val if b_neg else b_val

        # Mask reference number and compute HMAC fingerprint
        masked_ref = mask_reference(raw_ref)
        ref_fingerprint = None
        if raw_ref and normalize_reference(raw_ref):
            ref_fingerprint = compute_reference_fingerprint(raw_ref, hmac_key)

        # Date normalization
        iso_date, date_warning = normalize_date(raw_date)
        parsed_dt_date: Optional[dt_date] = None
        if date_warning:
            row_warnings.append(WarningItem(
                code=WarningCodes.AMBIGUOUS_DATE,
                message=date_warning,
                evidence_ids=[ev_id]
            ))
        elif iso_date:
            try:
                parsed_dt_date = dt_date.fromisoformat(iso_date)
            except ValueError:
                row_warnings.append(WarningItem(
                    code=WarningCodes.AMBIGUOUS_DATE,
                    message=f"Invalid date value: {iso_date}",
                    evidence_ids=[ev_id]
                ))
        else:
            row_warnings.append(WarningItem(
                code=WarningCodes.AMBIGUOUS_DATE,
                message="Missing transaction date.",
                evidence_ids=[ev_id]
            ))

        # Structured provenance (privacy-safe excerpt, no raw cell leakage)
        safe_snippet = f"Row {r_idx}: {sanitized_desc}"
        source_loc = SourceLocation(
            row_index=r_idx,
            raw_source_snippet=safe_snippet,
            description=source_loc_str
        )

        has_blocking_warning = any(w.code in (WarningCodes.VALUE_NOT_FOUND, WarningCodes.AMBIGUOUS_COLUMN, WarningCodes.AMBIGUOUS_DATE) for w in row_warnings)

        if txn_amount_dec is not None and txn_type is not None and parsed_dt_date is not None and not has_blocking_warning:
            txn = BankTransaction(
                id=ev_id,
                date=parsed_dt_date,
                description=sanitized_desc,
                amount=txn_amount_dec,
                type=TransactionType(txn_type),
                balance=norm_bal_dec,
                reference_number=masked_ref,
                source_document=filename,
                source_location=source_loc_str,
                warnings=[],
            )
            normalized_transactions.append(txn)

            amt_str = f"{txn_amount_dec:.2f}"
            val_str = f"{txn_type.upper()} {amt_str}"
            ev_rec = EvidenceRecord(
                id=ev_id,
                document_id=document_id,
                type=EvidenceType.REPAYMENT_TRANSACTION,
                value=val_str,
                source_document=filename,
                source_location=source_loc,
                confidence=0.95,
                verification_status="extracted",
                label="Bank Statement Transaction",
                normalized_value=amt_str,
                extraction_method="deterministic",
                confidence_basis="Parsed tabular CSV row",
                warnings=row_warnings,
                reference_number=masked_ref,
                reference_fingerprint=ref_fingerprint,
                amount=txn_amount_dec,
                date=parsed_dt_date,
                source_excerpt=safe_snippet
            )
        else:
            # Row was rejected: do NOT construct BankTransaction or fabricate zero-valued evidence!
            ev_rec = EvidenceRecord(
                id=ev_id,
                document_id=document_id,
                type=EvidenceType.FINANCIAL_MENTION,
                value=sanitized_desc if sanitized_desc and sanitized_desc.strip() else f"Row {r_idx}",
                source_document=filename,
                source_location=source_loc,
                confidence=0.0,
                verification_status="extracted",
                label="Bank Statement Transaction (Rejected)",
                normalized_value=None,
                extraction_method="deterministic",
                confidence_basis="Parsed tabular CSV row (rejected due to validation warnings)",
                warnings=row_warnings,
                reference_number=masked_ref,
                reference_fingerprint=ref_fingerprint,
                amount=None,
                date=parsed_dt_date,
                source_excerpt=safe_snippet
            )
        evidence_records.append(ev_rec)

    return normalized_transactions, evidence_records


def parse_txt_document(
    file_content: bytes,
    filename: str,
    document_id: str
) -> Tuple[str, List[EvidenceRecord]]:
    """
    Parses a plain-text document (such as KFS disclosures, loan agreements, or notes).
    Preserves raw line text internally and tracks line-number provenance.
    Emits raw text and initial line-level mentions.
    """
    try:
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            text = file_content.decode("latin-1")
    except Exception as e:
        raise CorruptedFileError(f"Could not decode text from '{filename}': {str(e)}")

    evidence_records: List[EvidenceRecord] = []
    lines = text.splitlines()
    loc_occurrences: Dict[str, int] = {}

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        loc_str = f"line {idx}"

        # Look for explicit loan amount mention: e.g. "Loan Amount: Rs. 50,000"
        loan_amt_match = re.search(r'(?:loan\s+amount|sanctioned\s+amount)[\s:]+(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)', stripped, flags=re.IGNORECASE)
        if loan_amt_match:
            raw_val = loan_amt_match.group(1)
            norm_amt = normalize_amount(raw_val)
            occ = loc_occurrences.get(loc_str, 0)
            loc_occurrences[loc_str] = occ + 1
            ev_id = make_evidence_id(document_id, loc_str, occ)

            # Character offsets within line
            c_start = loan_amt_match.start()
            c_end = loan_amt_match.end()
            sanitized_line = sanitize_free_text(stripped) or stripped

            source_loc = SourceLocation(
                line_number=idx,
                char_start=c_start,
                char_end=c_end,
                raw_source_snippet=sanitized_line,
                description=loc_str
            )

            evidence_records.append(EvidenceRecord(
                id=ev_id,
                document_id=document_id,
                type=EvidenceType.LOAN_AMOUNT,
                value=raw_val,
                source_document=filename,
                source_location=source_loc,
                confidence=0.9,
                verification_status="extracted",
                label="Sanctioned Loan Amount",
                normalized_value=str(norm_amt) if norm_amt else None,
                extraction_method="deterministic",
                confidence_basis="Regex matched KFS loan amount pattern",
                amount=norm_amt,
                source_excerpt=sanitized_line
            ))

        # Look for interest rate mention: e.g. "Interest Rate: 12.5% p.a."
        rate_match = re.search(r'(?:interest\s+rate|rate\s+of\s+interest|roi)[\s:]+([\d.]+\s*%)', stripped, flags=re.IGNORECASE)
        if rate_match:
            raw_rate = rate_match.group(1)
            occ = loc_occurrences.get(loc_str, 0)
            loc_occurrences[loc_str] = occ + 1
            ev_id = make_evidence_id(document_id, loc_str, occ)

            # Character offsets within line
            c_start = rate_match.start()
            c_end = rate_match.end()
            sanitized_line = sanitize_free_text(stripped) or stripped

            source_loc = SourceLocation(
                line_number=idx,
                char_start=c_start,
                char_end=c_end,
                raw_source_snippet=sanitized_line,
                description=loc_str
            )

            evidence_records.append(EvidenceRecord(
                id=ev_id,
                document_id=document_id,
                type=EvidenceType.INTEREST_RATE,
                value=raw_rate,
                source_document=filename,
                source_location=source_loc,
                confidence=0.9,
                verification_status="extracted",
                label="Annual Interest Rate",
                normalized_value=raw_rate.replace("%", "").strip(),
                extraction_method="deterministic",
                confidence_basis="Regex matched KFS interest rate pattern",
                source_excerpt=sanitized_line
            ))

    return text, evidence_records


def parse_pdf_document(
    file_content: bytes,
    filename: str,
    document_id: str,
    hmac_key: Optional[bytes] = None
) -> Tuple[List[EvidenceRecord], List[WarningItem]]:
    """
    Parses a text-based PDF document page by page.
    Preserves page-level source location ('Page 1', 'Page 2', etc.) and deterministic evidence IDs.
    Extracts financial mentions (loan amount, interest rate, APR, tenure, EMI, fees, references)
    without inventing or defaulting missing values.

    Explicit failure modes:
    - Encrypted PDF -> EncryptedPDFError
    - Malformed/unreadable PDF -> CorruptedFileError
    - Genuinely empty PDF -> EmptyDocumentError
    - Scanned/image-only PDF without extractable text layer -> UnsupportedScannedPDFError
    """
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
    except Exception as e:
        raise CorruptedFileError(f"Could not open or parse PDF document '{filename}': {str(e)}")

    if doc.is_encrypted:
        doc.close()
        raise EncryptedPDFError(f"PDF document '{filename}' is password-protected or encrypted.")

    if doc.page_count == 0:
        doc.close()
        raise EmptyDocumentError(f"PDF document '{filename}' contains no pages.")

    total_text = ""
    total_images = 0
    pages_data = []

    for page_idx in range(doc.page_count):
        try:
            page = doc.load_page(page_idx)
            text = page.get_text() or ""
            images = page.get_images() or []
        except Exception as e:
            doc.close()
            raise CorruptedFileError(f"Error reading page {page_idx + 1} of PDF '{filename}': {str(e)}")

        total_text += text
        total_images += len(images)
        pages_data.append((page_idx + 1, text))

    doc.close()

    if not total_text.strip():
        if total_images > 0:
            raise UnsupportedScannedPDFError(
                f"PDF document '{filename}' appears to be a scanned or image-only document with no extractable text layer."
            )
        else:
            raise EmptyDocumentError(f"PDF document '{filename}' contains no extractable text or content.")

    evidence_records: List[EvidenceRecord] = []
    pdf_warnings: List[WarningItem] = []
    loc_occurrences: Dict[str, int] = {}

    for page_num, page_text in pages_data:
        loc_str = f"Page {page_num}"
        lines = page_text.splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            sanitized_line = sanitize_free_text(stripped) or stripped

            # 1. Loan Amount / Sanctioned Amount
            loan_match = re.search(
                r'(?:loan\s+amount|sanctioned\s+amount|principal\s+amount)[\s:]+(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)',
                stripped,
                flags=re.IGNORECASE
            )
            if loan_match:
                raw_val = loan_match.group(1)
                norm_amt = normalize_amount(raw_val)
                occ = loc_occurrences.get(loc_str, 0)
                loc_occurrences[loc_str] = occ + 1
                ev_id = make_evidence_id(document_id, loc_str, occ)
                source_loc = SourceLocation(
                    page_number=page_num,
                    raw_source_snippet=sanitized_line,
                    description=loc_str
                )
                evidence_records.append(EvidenceRecord(
                    id=ev_id,
                    document_id=document_id,
                    type=EvidenceType.LOAN_AMOUNT,
                    value=raw_val,
                    source_document=filename,
                    source_location=source_loc,
                    confidence=0.9,
                    verification_status=VerificationStatus.EXTRACTED,
                    label="Sanctioned Loan Amount",
                    normalized_value=str(norm_amt) if norm_amt else None,
                    extraction_method="deterministic",
                    confidence_basis="Regex matched PDF loan amount pattern",
                    amount=norm_amt,
                    source_excerpt=sanitized_line
                ))

            # 2. Net Disbursement
            net_match = re.search(
                r'(?:net\s+disbursement|disbursed\s+amount|net\s+disbursal)[\s:]+(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)',
                stripped,
                flags=re.IGNORECASE
            )
            if net_match:
                raw_val = net_match.group(1)
                norm_amt = normalize_amount(raw_val)
                occ = loc_occurrences.get(loc_str, 0)
                loc_occurrences[loc_str] = occ + 1
                ev_id = make_evidence_id(document_id, loc_str, occ)
                source_loc = SourceLocation(
                    page_number=page_num,
                    raw_source_snippet=sanitized_line,
                    description=loc_str
                )
                evidence_records.append(EvidenceRecord(
                    id=ev_id,
                    document_id=document_id,
                    type=EvidenceType.NET_DISBURSEMENT,
                    value=raw_val,
                    source_document=filename,
                    source_location=source_loc,
                    confidence=0.9,
                    verification_status=VerificationStatus.EXTRACTED,
                    label="Net Disbursement Amount",
                    normalized_value=str(norm_amt) if norm_amt else None,
                    extraction_method="deterministic",
                    confidence_basis="Regex matched PDF net disbursement pattern",
                    amount=norm_amt,
                    source_excerpt=sanitized_line
                ))

            # 3. Disclosed APR
            apr_match = re.search(
                r'(?:annual\s+percentage\s+rate|disclosed\s+apr|apr)[\s:]+([\d.]+\s*%)',
                stripped,
                flags=re.IGNORECASE
            )
            if apr_match:
                raw_rate = apr_match.group(1)
                norm_pct = normalize_percentage(raw_rate)
                occ = loc_occurrences.get(loc_str, 0)
                loc_occurrences[loc_str] = occ + 1
                ev_id = make_evidence_id(document_id, loc_str, occ)
                source_loc = SourceLocation(
                    page_number=page_num,
                    raw_source_snippet=sanitized_line,
                    description=loc_str
                )
                evidence_records.append(EvidenceRecord(
                    id=ev_id,
                    document_id=document_id,
                    type=EvidenceType.DISCLOSED_APR,
                    value=raw_rate,
                    source_document=filename,
                    source_location=source_loc,
                    confidence=0.9,
                    verification_status=VerificationStatus.EXTRACTED,
                    label="Disclosed APR",
                    normalized_value=str(norm_pct) if norm_pct else None,
                    extraction_method="deterministic",
                    confidence_basis="Regex matched PDF disclosed APR pattern",
                    source_excerpt=sanitized_line
                ))

            # 4. Interest Rate
            rate_match = re.search(
                r'(?:interest\s+rate|rate\s+of\s+interest|roi)[\s:]+([\d.]+\s*%)',
                stripped,
                flags=re.IGNORECASE
            )
            if rate_match and not apr_match:
                raw_rate = rate_match.group(1)
                norm_pct = normalize_percentage(raw_rate)
                occ = loc_occurrences.get(loc_str, 0)
                loc_occurrences[loc_str] = occ + 1
                ev_id = make_evidence_id(document_id, loc_str, occ)
                source_loc = SourceLocation(
                    page_number=page_num,
                    raw_source_snippet=sanitized_line,
                    description=loc_str
                )
                evidence_records.append(EvidenceRecord(
                    id=ev_id,
                    document_id=document_id,
                    type=EvidenceType.INTEREST_RATE,
                    value=raw_rate,
                    source_document=filename,
                    source_location=source_loc,
                    confidence=0.9,
                    verification_status=VerificationStatus.EXTRACTED,
                    label="Annual Interest Rate",
                    normalized_value=str(norm_pct) if norm_pct else None,
                    extraction_method="deterministic",
                    confidence_basis="Regex matched PDF interest rate pattern",
                    source_excerpt=sanitized_line
                ))

            # 5. EMI Amount
            emi_match = re.search(
                r'(?:emi\s+amount|monthly\s+instalment|monthly\s+installment|emi)[\s:]+(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)',
                stripped,
                flags=re.IGNORECASE
            )
            if emi_match:
                raw_val = emi_match.group(1)
                norm_amt = normalize_amount(raw_val)
                occ = loc_occurrences.get(loc_str, 0)
                loc_occurrences[loc_str] = occ + 1
                ev_id = make_evidence_id(document_id, loc_str, occ)
                source_loc = SourceLocation(
                    page_number=page_num,
                    raw_source_snippet=sanitized_line,
                    description=loc_str
                )
                evidence_records.append(EvidenceRecord(
                    id=ev_id,
                    document_id=document_id,
                    type=EvidenceType.EMI_AMOUNT,
                    value=raw_val,
                    source_document=filename,
                    source_location=source_loc,
                    confidence=0.9,
                    verification_status=VerificationStatus.EXTRACTED,
                    label="Monthly EMI Amount",
                    normalized_value=str(norm_amt) if norm_amt else None,
                    extraction_method="deterministic",
                    confidence_basis="Regex matched PDF EMI amount pattern",
                    amount=norm_amt,
                    source_excerpt=sanitized_line
                ))

            # 6. Tenure Months
            tenure_match = re.search(
                r'(?:tenure|loan\s+period|loan\s+term|duration)[\s:]+(\d+)\s*(?:months|mths|m)\b',
                stripped,
                flags=re.IGNORECASE
            )
            if tenure_match:
                raw_val = tenure_match.group(1)
                occ = loc_occurrences.get(loc_str, 0)
                loc_occurrences[loc_str] = occ + 1
                ev_id = make_evidence_id(document_id, loc_str, occ)
                source_loc = SourceLocation(
                    page_number=page_num,
                    raw_source_snippet=sanitized_line,
                    description=loc_str
                )
                evidence_records.append(EvidenceRecord(
                    id=ev_id,
                    document_id=document_id,
                    type=EvidenceType.TENURE_MONTHS,
                    value=raw_val,
                    source_document=filename,
                    source_location=source_loc,
                    confidence=0.9,
                    verification_status=VerificationStatus.EXTRACTED,
                    label="Loan Tenure (Months)",
                    normalized_value=raw_val,
                    extraction_method="deterministic",
                    confidence_basis="Regex matched PDF tenure pattern",
                    source_excerpt=sanitized_line
                ))

            # 7. First EMI Date
            date_match = re.search(
                r'(?:first\s+emi\s+date|repayment\s+start\s+date)[\s:]+([0-9a-zA-Z/.-]+)',
                stripped,
                flags=re.IGNORECASE
            )
            if date_match:
                raw_date_str = date_match.group(1).strip()
                iso_date, date_warn = normalize_date(raw_date_str)
                occ = loc_occurrences.get(loc_str, 0)
                loc_occurrences[loc_str] = occ + 1
                ev_id = make_evidence_id(document_id, loc_str, occ)
                source_loc = SourceLocation(
                    page_number=page_num,
                    raw_source_snippet=sanitized_line,
                    description=loc_str
                )
                parsed_d = None
                date_warnings: List[WarningItem] = []
                if date_warn:
                    w = WarningItem(code=WarningCodes.AMBIGUOUS_DATE, message=date_warn, evidence_ids=[ev_id])
                    date_warnings.append(w)
                    pdf_warnings.append(w)
                elif iso_date:
                    try:
                        parsed_d = dt_date.fromisoformat(iso_date)
                    except ValueError:
                        w = WarningItem(code=WarningCodes.AMBIGUOUS_DATE, message=f"Invalid date: {iso_date}", evidence_ids=[ev_id])
                        date_warnings.append(w)
                        pdf_warnings.append(w)

                evidence_records.append(EvidenceRecord(
                    id=ev_id,
                    document_id=document_id,
                    type=EvidenceType.FIRST_EMI_DATE,
                    value=raw_date_str,
                    source_document=filename,
                    source_location=source_loc,
                    confidence=0.9 if parsed_d else 0.5,
                    verification_status=VerificationStatus.EXTRACTED,
                    label="First EMI Repayment Date",
                    normalized_value=iso_date if parsed_d else None,
                    extraction_method="deterministic",
                    confidence_basis="Regex matched PDF first EMI date pattern",
                    date=parsed_d,
                    warnings=date_warnings,
                    source_excerpt=sanitized_line
                ))

            # 8. Processing Fee
            fee_match = re.search(
                r'(?:processing\s+fee|processing\s+charges?)[\s:]+([^\n]+)',
                stripped,
                flags=re.IGNORECASE
            )
            if fee_match:
                raw_fee = fee_match.group(1).strip()
                occ = loc_occurrences.get(loc_str, 0)
                loc_occurrences[loc_str] = occ + 1
                ev_id = make_evidence_id(document_id, loc_str, occ)
                source_loc = SourceLocation(
                    page_number=page_num,
                    raw_source_snippet=sanitized_line,
                    description=loc_str
                )
                fee_model = parse_processing_fee(raw_fee, [ev_id])
                norm_fee_str = str(fee_model.value) if fee_model and fee_model.value is not None else None
                evidence_records.append(EvidenceRecord(
                    id=ev_id,
                    document_id=document_id,
                    type=EvidenceType.PROCESSING_FEE,
                    value=raw_fee,
                    source_document=filename,
                    source_location=source_loc,
                    confidence=0.9,
                    verification_status=VerificationStatus.EXTRACTED,
                    label="Processing Fee",
                    normalized_value=norm_fee_str,
                    extraction_method="deterministic",
                    confidence_basis="Regex matched PDF processing fee pattern",
                    amount=fee_model.value if fee_model else None,
                    currency=fee_model.currency if fee_model else None,
                    warnings=fee_model.warnings if fee_model else [],
                    source_excerpt=sanitized_line
                ))

            # 9. Reference numbers (Account number / Loan reference)
            ref_match = re.search(
                r'(?i)\b(loan\s+(?:id|ref(?:erence)?|account|a/c)|account\s+(?:no|number)|a/c)\b\s*[:#-]?\s*([A-Za-z0-9/-]+)',
                stripped
            )
            if ref_match:
                raw_ref = ref_match.group(2).strip()
                if normalize_reference(raw_ref):
                    masked_ref = mask_reference(raw_ref)
                    ref_fp = compute_reference_fingerprint(raw_ref, hmac_key)
                    occ = loc_occurrences.get(loc_str, 0)
                    loc_occurrences[loc_str] = occ + 1
                    ev_id = make_evidence_id(document_id, loc_str, occ)
                    source_loc = SourceLocation(
                        page_number=page_num,
                        raw_source_snippet=sanitized_line,
                        description=loc_str
                    )
                    is_loan = "loan" in ref_match.group(1).lower()
                    ev_type = EvidenceType.LOAN_MENTION if is_loan else EvidenceType.BANK_ACCOUNT_MENTION
                    evidence_records.append(EvidenceRecord(
                        id=ev_id,
                        document_id=document_id,
                        type=ev_type,
                        value=sanitized_line,
                        source_document=filename,
                        source_location=source_loc,
                        confidence=0.9,
                        verification_status=VerificationStatus.EXTRACTED,
                        label="Loan Reference Mention" if is_loan else "Bank Account Mention",
                        normalized_value=masked_ref,
                        extraction_method="deterministic",
                        confidence_basis="Regex matched PDF account/reference pattern",
                        reference_number=masked_ref,
                        reference_fingerprint=ref_fp,
                        source_excerpt=sanitized_line
                    ))

    return evidence_records, pdf_warnings


def extract_document(
    file_content: bytes,
    filename: str,
    document_id: str,
    hmac_key: Optional[bytes] = None
) -> ExtractionResult:
    """
    Ingests and extracts evidence, transactions, and structured warnings from a document.
    Emits a canonical ExtractionResult for orchestration.
    """
    validation = validate_file_upload(file_content, filename)
    file_type = validation.file_type

    all_evidence: List[EvidenceRecord] = []
    all_transactions: List[BankTransaction] = []
    all_warnings: List[WarningItem] = []

    if file_type == "csv":
        txns, evidence = parse_csv_transactions(
            file_content=file_content,
            filename=filename,
            document_id=document_id,
            hmac_key=hmac_key
        )
        all_transactions.extend(txns)
        all_evidence.extend(evidence)
        # Aggregate structured warnings from all evidence records (without duplicates)
        seen_warn_keys = set()
        for ev in evidence:
            for w in ev.warnings:
                key = (w.code, w.message, tuple(w.evidence_ids))
                if key not in seen_warn_keys:
                    seen_warn_keys.add(key)
                    all_warnings.append(w)
    elif file_type == "txt":
        raw_text, evidence = parse_txt_document(
            file_content=file_content,
            filename=filename,
            document_id=document_id
        )
        all_evidence.extend(evidence)
        for ev in evidence:
            all_warnings.extend(ev.warnings)
    elif file_type == "pdf":
        evidence, pdf_warnings = parse_pdf_document(
            file_content=file_content,
            filename=filename,
            document_id=document_id,
            hmac_key=hmac_key,
        )
        all_evidence.extend(evidence)
        all_warnings.extend(pdf_warnings)
        seen_warn_keys = set((w.code, w.message, tuple(w.evidence_ids)) for w in all_warnings)
        for ev in evidence:
            for w in ev.warnings:
                key = (w.code, w.message, tuple(w.evidence_ids))
                if key not in seen_warn_keys:
                    seen_warn_keys.add(key)
                    all_warnings.append(w)

    return ExtractionResult(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        evidence=all_evidence,
        transactions=all_transactions,
        warnings=all_warnings,
        metadata={"file_size_bytes": validation.file_size_bytes}
    )


def verify_evidence(
    record: EvidenceRecord,
    verification_source: str,
    target_status: VerificationStatus = VerificationStatus.VERIFIED,
    notes: Optional[str] = None
) -> EvidenceRecord:
    """
    Dedicated verification transition service.
    Enforces that evidence records can only transition to 'verified' with an explicit,
    non-empty verification source (such as bank confirmation, cross-source match,
    or independent ledger).
    Sensitive references in verification sources or notes are sanitized.
    """
    if target_status == VerificationStatus.VERIFIED:
        if not verification_source or not str(verification_source).strip():
            raise ValueError(
                "Verification source is required to mark evidence as verified. "
                "Confidence or extraction alone cannot justify verified status."
            )
        # Cannot verify if already conflicting or unverified without resolution
        if record.verification_status == VerificationStatus.CONFLICTING:
            raise ValueError("Conflicting evidence cannot be marked as verified without resolving the conflict.")

    safe_source = sanitize_free_text(verification_source) or "Unknown verification source"
    safe_notes = sanitize_free_text(notes) if notes else None

    basis_parts = []
    if record.confidence_basis:
        basis_parts.append(record.confidence_basis)
    basis_parts.append(f"Verified via: {safe_source}")
    if safe_notes:
        basis_parts.append(f"Notes: {safe_notes}")
    new_basis = " | ".join(basis_parts)

    dumped = record.model_dump()
    dumped["verification_status"] = target_status.value if isinstance(target_status, VerificationStatus) else target_status
    dumped["confidence_basis"] = new_basis

    # Reconstruct preserving internal structured source location
    updated = EvidenceRecord(**dumped)
    if record.structured_source_location is not None:
        updated._source_location_obj = record.structured_source_location

    return updated
