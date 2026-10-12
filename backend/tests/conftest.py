import pytest
import os
import sys
from pathlib import Path

# Add backend directory to sys.path so 'import app.schemas' works seamlessly
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

@pytest.fixture(autouse=True)
def set_test_hmac_key(monkeypatch):
    """Ensure a safe test HMAC key is present in environment for tests."""
    monkeypatch.setenv("PROOFLEDGER_REFERENCE_HMAC_KEY", "test-secret-key-32-chars-long!")
