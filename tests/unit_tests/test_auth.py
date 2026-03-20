"""Unit tests for authentication utilities."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import mock_open, patch

import pytest

from langchain_githubcopilot_chat.auth import (
    _sync_token_refresh_lock,
    load_tokens_from_cache,
    save_tokens_to_cache,
)

# ---------------------------------------------------------------------------
# Threading lock type
# ---------------------------------------------------------------------------


def test_sync_lock_is_threading_lock() -> None:
    """The module-level sync lock must be a real threading.Lock (not a bool)."""
    assert isinstance(_sync_token_refresh_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# save_tokens_to_cache
# ---------------------------------------------------------------------------


def test_save_tokens_logs_on_oserror(caplog: pytest.LogCaptureFixture) -> None:
    """OSError during cache save should be logged as a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="langchain_githubcopilot_chat.auth"):
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            save_tokens_to_cache("ghp_tok", "copilot_tok")

    assert any("Failed to save" in r.message for r in caplog.records)


def test_save_tokens_does_not_raise_on_oserror() -> None:
    """OSError during cache save must NOT propagate."""
    with patch("builtins.open", side_effect=OSError("disk full")):
        save_tokens_to_cache("ghp_tok", "copilot_tok")  # should not raise


# ---------------------------------------------------------------------------
# load_tokens_from_cache
# ---------------------------------------------------------------------------


def test_load_tokens_returns_empty_on_file_not_found() -> None:
    """Missing cache file should return {} silently."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        result = load_tokens_from_cache()
    assert result == {}


def test_load_tokens_logs_on_corrupt_json(caplog: pytest.LogCaptureFixture) -> None:
    """Corrupt JSON in the cache file should be logged as a warning."""
    import logging

    bad_json = "not valid json {"
    with caplog.at_level(logging.WARNING, logger="langchain_githubcopilot_chat.auth"):
        with patch("builtins.open", mock_open(read_data=bad_json)):
            result = load_tokens_from_cache()

    assert result == {}
    assert any("Failed to load" in r.message for r in caplog.records)


def test_load_tokens_returns_empty_on_expired_token(tmp_path: object) -> None:
    """A cached token past its expires_at should return {}."""
    expired_data = {
        "github_token": "ghp_tok",
        "copilot_token": "copilot_old",
        "expires_at": time.time() - 10,
    }
    with patch("builtins.open", mock_open(read_data=json.dumps(expired_data))):
        result = load_tokens_from_cache()
    assert result == {}


def test_load_tokens_returns_data_for_valid_token() -> None:
    """A non-expired cached token should be returned."""
    valid_data = {
        "github_token": "ghp_tok",
        "copilot_token": "copilot_good",
        "expires_at": time.time() + 3600,
    }
    with patch("builtins.open", mock_open(read_data=json.dumps(valid_data))):
        result = load_tokens_from_cache()
    assert result["copilot_token"] == "copilot_good"
