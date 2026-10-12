"""Domain exceptions for ProofLedger document intelligence layer."""


class ProofLedgerDocumentError(Exception):
    """Base exception for all document intelligence operations."""
    pass


class IngestionError(ProofLedgerDocumentError):
    """Raised when document ingestion or file parsing fails."""
    pass


class FileValidationError(IngestionError):
    """Raised when file exceeds size limits or has an unsupported format."""
    pass


class FileSizeLimitExceededError(FileValidationError):
    """Raised when file size exceeds the 10MB limit."""
    pass


class UnsupportedFileTypeError(FileValidationError):
    """Raised when file extension or magic bytes do not match PDF, CSV, or TXT."""
    pass


class CorruptedFileError(IngestionError):
    """Raised when file bytes cannot be parsed as valid text or structure."""
    pass


class EncryptedPDFError(IngestionError):
    """Raised when a PDF file is password-protected or encrypted."""
    pass


class EmptyDocumentError(IngestionError):
    """Raised when a document contains no pages or no extractable content."""
    pass


class UnsupportedScannedPDFError(IngestionError):
    """Raised when a PDF file contains scanned images without an extractable text layer."""
    pass


class NormalizationError(ProofLedgerDocumentError):
    """Base error for normalizers."""
    pass


class AmbiguousDateError(NormalizationError):
    """Raised or flagged when date representation cannot be resolved unambiguously."""
    pass


class InvalidNumericFormatError(NormalizationError):
    """Raised when numeric string cannot be parsed deterministically."""
    pass


class ExtractionError(ProofLedgerDocumentError):
    """Raised when extraction fails."""
    pass


class MissingHMACKeyError(ProofLedgerDocumentError):
    """Raised when PROOFLEDGER_REFERENCE_HMAC_KEY is required but not configured."""
    pass
