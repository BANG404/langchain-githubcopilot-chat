"""Unit tests for ChatGithubCopilot chat model (ChatOpenAI-based implementation)."""

from __future__ import annotations

import threading
import time
from typing import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

import langchain_githubcopilot_chat.auth as _auth_module
from langchain_githubcopilot_chat.chat_models import (
    _GITHUB_COPILOT_BASE_URL,
    _TOKEN_REFRESH_BUFFER_SECS,
    ChatGithubCopilot,
    ChatGithubcopilotChat,
    _is_auth_error,
    _is_exchangeable_github_token,
)

# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------


def test_alias_is_same_class() -> None:
    """ChatGithubcopilotChat should be the same class as ChatGithubCopilot."""
    assert ChatGithubcopilotChat is ChatGithubCopilot


# ---------------------------------------------------------------------------
# Instantiation & field resolution
# ---------------------------------------------------------------------------


def test_instantiation_with_explicit_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly supplied token should take priority over GITHUB_TOKEN env var."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value={},
        ),
        patch(
            "langchain_githubcopilot_chat.chat_models.fetch_copilot_token",
            return_value=("tid_fake", None),
        ),
        patch("langchain_githubcopilot_chat.chat_models.save_tokens_to_cache"),
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="ghp_testtoken")
    assert llm.model_name == "openai/gpt-4.1"
    assert llm.github_token is not None
    assert llm.github_token.get_secret_value() == "ghp_testtoken"


def test_instantiation_with_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """GITHUB_TOKEN env var should be used when no explicit token is provided."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_envtoken")
    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value={},
        ),
        patch(
            "langchain_githubcopilot_chat.chat_models.fetch_copilot_token",
            return_value=("tid_fake", None),
        ),
        patch("langchain_githubcopilot_chat.chat_models.save_tokens_to_cache"),
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1")
    assert llm.github_token is not None
    assert llm.github_token.get_secret_value() == "ghp_envtoken"


def test_instantiation_with_cached_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache fallback: github_token is loaded from ~/.github-copilot-chat.json."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    future_exp = time.time() + 7200
    cache = {
        "github_token": "gho_cached_gh",
        "copilot_token": "copilot_cached",
        "expires_at": future_exp,
    }
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value=cache,
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1")
    assert llm.github_token is not None
    assert llm.github_token.get_secret_value() == "gho_cached_gh"


def test_instantiation_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing token should raise a ValueError."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value={},
    ):
        with pytest.raises(ValueError, match="GitHub token is required"):
            ChatGithubCopilot(model="openai/gpt-4.1")


def test_openai_api_key_set_to_copilot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a cached copilot token exists, openai_api_key should be set to it."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    future_exp = time.time() + 7200
    cache = {
        "github_token": "gho_gh",
        "copilot_token": "copilot_xyz",
        "expires_at": future_exp,
    }
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value=cache,
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1")
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "copilot_xyz"


def test_openai_api_base_is_copilot_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """openai_api_base should point to the GitHub Copilot API."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value={},
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="fake-tok")
    assert _GITHUB_COPILOT_BASE_URL in str(llm.openai_api_base)


def test_copilot_headers_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Copilot-required headers should be present in default_headers."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value={},
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="fake-tok")
    assert "Copilot-Integration-Id" in llm.default_headers


def test_model_name() -> None:
    """model_name should reflect the model passed at init."""
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value={},
    ):
        llm = ChatGithubCopilot(model="meta/llama-3.3-70b-instruct", github_token="tok")
    assert llm.model_name == "meta/llama-3.3-70b-instruct"


def test_llm_type() -> None:
    """_llm_type should return 'github-copilot'."""
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value={},
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    assert llm._llm_type == "github-copilot"


def test_github_token_always_stored_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """github_token must be persisted even when sourced from the file cache."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    future_exp = time.time() + 7200
    cache = {
        "github_token": "gho_from_cache",
        "copilot_token": "cp_tok",
        "expires_at": future_exp,
    }
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value=cache,
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1")
    # Must be stored so _refresh_copilot_token can exchange it later
    assert llm.github_token is not None
    assert llm.github_token.get_secret_value() == "gho_from_cache"


# ---------------------------------------------------------------------------
# Token refresh helpers
# ---------------------------------------------------------------------------


def _make_llm(**kwargs) -> ChatGithubCopilot:
    """Create a ChatGithubCopilot with a mocked cache and no exchange."""
    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value={},
        ),
        patch(
            "langchain_githubcopilot_chat.chat_models.fetch_copilot_token",
            return_value=("tid_fake_copilot_token", None),
        ),
        patch("langchain_githubcopilot_chat.chat_models.save_tokens_to_cache"),
    ):
        return ChatGithubCopilot(
            model="openai/gpt-4.1", github_token="gho_gh", **kwargs
        )


def test_get_github_token_str_from_field() -> None:
    llm = _make_llm()
    assert llm._get_github_token_str() == "gho_gh"


def test_get_github_token_str_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value={},
        ),
        patch(
            "langchain_githubcopilot_chat.chat_models.fetch_copilot_token",
            return_value=("tid_fake", None),
        ),
        patch("langchain_githubcopilot_chat.chat_models.save_tokens_to_cache"),
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="gho_explicit")
    # Clear field to force env fallback
    object.__setattr__(llm, "github_token", None)
    monkeypatch.setenv("GITHUB_TOKEN", "gho_env")
    assert llm._get_github_token_str() == "gho_env"


def test_refresh_copilot_token_updates_api_key() -> None:
    """_refresh_copilot_token should update openai_api_key and rebuild clients."""
    llm = _make_llm()

    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.fetch_copilot_token",
            return_value=("new_copilot_tok", time.time() + 3600),
        ),
        patch("langchain_githubcopilot_chat.chat_models.save_tokens_to_cache"),
        patch.object(llm, "_rebuild_clients"),
    ):
        result = llm._refresh_copilot_token()

    assert result is True
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "new_copilot_tok"


def test_refresh_copilot_token_returns_false_without_oauth_token() -> None:
    """_refresh_copilot_token returns False when no gho_/ghp_/ghu_ token is available."""  # noqa: E501
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value={},
    ):
        llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="non-oauth-token")

    result = llm._refresh_copilot_token()
    assert result is False


def test_refresh_copilot_token_returns_false_when_exchange_fails() -> None:
    """_refresh_copilot_token returns False when the token exchange returns None."""
    llm = _make_llm()
    with patch(
        "langchain_githubcopilot_chat.chat_models.fetch_copilot_token",
        return_value=(None, None),
    ):
        result = llm._refresh_copilot_token()
    assert result is False


def test_refresh_copilot_token_sync_skips_when_lock_held() -> None:
    """If the sync lock is held, _refresh_copilot_token should not block indefinitely."""  # noqa: E501
    llm = _make_llm()
    original_key = llm.openai_api_key

    _auth_module._sync_token_refresh_lock.acquire(blocking=True)
    try:
        # Lock is held — refresh should detect non-blocking acquire failed, wait, then return  # noqa: E501
        # We run in a thread so the main test doesn't deadlock
        results = []

        def do_refresh():
            # The non-blocking acquire fails → waits for blocking acquire, then returns False  # noqa: E501
            # Because nothing changes (no fetch is called with a held lock)
            r = llm._refresh_copilot_token()
            results.append(r)

        t = threading.Thread(target=do_refresh)
        t.start()
        # Give the thread a moment to try acquiring
        import time as _time

        _time.sleep(0.1)
    finally:
        _auth_module._sync_token_refresh_lock.release()

    t.join(timeout=2)
    # The token should not have changed (no fetch happened)
    assert llm.openai_api_key == original_key


@pytest.mark.asyncio
async def test_arefresh_copilot_token_updates_api_key() -> None:
    """_arefresh_copilot_token should update openai_api_key."""
    llm = _make_llm()

    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.afetch_copilot_token",
            new_callable=AsyncMock,
            return_value=("async_new_tok", time.time() + 3600),
        ),
        patch("langchain_githubcopilot_chat.chat_models.save_tokens_to_cache"),
        patch.object(llm, "_rebuild_clients"),
    ):
        result = await llm._arefresh_copilot_token()

    assert result is True
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "async_new_tok"


# ---------------------------------------------------------------------------
# Proactive token expiry check
# ---------------------------------------------------------------------------


def test_maybe_refresh_proactively_triggers_when_near_expiry() -> None:
    """_maybe_refresh_token_proactively should refresh when token is near/past expiry."""  # noqa: E501
    llm = _make_llm()
    near_exp = time.time() + (_TOKEN_REFRESH_BUFFER_SECS - 10)
    cache = {"expires_at": near_exp}

    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value=cache,
        ),
        patch.object(llm, "_refresh_copilot_token") as mock_refresh,
    ):
        llm._maybe_refresh_token_proactively()

    mock_refresh.assert_called_once()


def test_maybe_refresh_proactively_skips_when_token_fresh() -> None:
    """_maybe_refresh_token_proactively should not refresh a fresh token."""
    llm = _make_llm()
    far_exp = time.time() + 7200
    cache = {"expires_at": far_exp}

    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value=cache,
        ),
        patch.object(llm, "_refresh_copilot_token") as mock_refresh,
    ):
        llm._maybe_refresh_token_proactively()

    mock_refresh.assert_not_called()


@pytest.mark.asyncio
async def test_amaybe_refresh_proactively_triggers() -> None:
    """Async version should trigger refresh when token near/past expiry."""
    llm = _make_llm()
    near_exp = time.time() + (_TOKEN_REFRESH_BUFFER_SECS - 10)
    cache = {"expires_at": near_exp}

    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value=cache,
        ),
        patch.object(
            llm, "_arefresh_copilot_token", new_callable=AsyncMock
        ) as mock_refresh,
    ):
        await llm._amaybe_refresh_token_proactively()

    mock_refresh.assert_called_once()


# ---------------------------------------------------------------------------
# _generate override: 401 retry
# ---------------------------------------------------------------------------


def _make_auth_error() -> openai.AuthenticationError:
    resp = MagicMock()
    resp.status_code = 401
    resp.headers = {}
    return openai.AuthenticationError("token expired", response=resp, body={})


def _make_bad_request_auth_error() -> openai.BadRequestError:
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {}
    return openai.BadRequestError(
        "bad request: Authorization header is badly formatted",
        response=resp,
        body={},
    )


def test_generate_retries_on_auth_error() -> None:
    """_generate should catch AuthenticationError, refresh token, and retry once."""
    llm = _make_llm()
    call_count = [0]

    def mock_parent_generate(self, messages, stop=None, run_manager=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise _make_auth_error()
        return MagicMock(spec=ChatResult)

    with (
        patch.object(llm, "_refresh_copilot_token", return_value=True) as mock_refresh,
        patch.object(llm, "_maybe_refresh_token_proactively"),
        patch("langchain_openai.ChatOpenAI._generate", mock_parent_generate),
    ):
        result = llm._generate([HumanMessage("hi")])

    mock_refresh.assert_called_once()
    assert call_count[0] == 2
    assert result is not None


def test_generate_raises_if_refresh_fails() -> None:
    """_generate should re-raise AuthenticationError if refresh returns False."""
    llm = _make_llm()

    def mock_parent_generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise _make_auth_error()

    with (
        patch.object(llm, "_refresh_copilot_token", return_value=False),
        patch.object(llm, "_maybe_refresh_token_proactively"),
        patch("langchain_openai.ChatOpenAI._generate", mock_parent_generate),
    ):
        with pytest.raises(openai.AuthenticationError):
            llm._generate([HumanMessage("hi")])


def test_generate_retries_on_badly_formatted_auth_header() -> None:
    """_generate should also catch 400 BadRequestError with Authorization message."""
    llm = _make_llm()
    call_count = [0]

    def mock_parent_generate(self, messages, stop=None, run_manager=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise _make_bad_request_auth_error()
        return MagicMock(spec=ChatResult)

    with (
        patch.object(llm, "_refresh_copilot_token", return_value=True) as mock_refresh,
        patch.object(llm, "_maybe_refresh_token_proactively"),
        patch("langchain_openai.ChatOpenAI._generate", mock_parent_generate),
    ):
        llm._generate([HumanMessage("hi")])

    mock_refresh.assert_called_once()
    assert call_count[0] == 2


def test_generate_proactive_refresh_called() -> None:
    """_generate should call _maybe_refresh_token_proactively before each request."""
    llm = _make_llm()
    fake_result = MagicMock(spec=ChatResult)

    def mock_parent_generate(self, messages, stop=None, run_manager=None, **kwargs):
        return fake_result

    with (
        patch.object(llm, "_maybe_refresh_token_proactively") as mock_proactive,
        patch("langchain_openai.ChatOpenAI._generate", mock_parent_generate),
    ):
        llm._generate([HumanMessage("hi")])

    mock_proactive.assert_called_once()


# ---------------------------------------------------------------------------
# _stream override: 401 retry
# ---------------------------------------------------------------------------


def _make_chunk() -> ChatGenerationChunk:
    from langchain_core.messages import AIMessageChunk

    return ChatGenerationChunk(message=AIMessageChunk(content="hello"))


def test_stream_retries_on_auth_error() -> None:
    """_stream should refresh token and retry on AuthenticationError."""
    llm = _make_llm()
    call_count = [0]

    def mock_parent_stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator:
        call_count[0] += 1
        if call_count[0] == 1:
            raise _make_auth_error()
        yield _make_chunk()

    with (
        patch.object(llm, "_refresh_copilot_token", return_value=True) as mock_refresh,
        patch.object(llm, "_maybe_refresh_token_proactively"),
        patch("langchain_openai.ChatOpenAI._stream", mock_parent_stream),
    ):
        chunks = list(llm._stream([HumanMessage("hi")]))

    mock_refresh.assert_called_once()
    assert len(chunks) == 1


def test_stream_raises_if_refresh_fails() -> None:
    """_stream should re-raise AuthenticationError if refresh returns False."""
    llm = _make_llm()

    def mock_parent_stream(self, messages, stop=None, run_manager=None, **kwargs):
        raise _make_auth_error()
        yield  # make it a generator

    with (
        patch.object(llm, "_refresh_copilot_token", return_value=False),
        patch.object(llm, "_maybe_refresh_token_proactively"),
        patch("langchain_openai.ChatOpenAI._stream", mock_parent_stream),
    ):
        with pytest.raises(openai.AuthenticationError):
            list(llm._stream([HumanMessage("hi")]))


# ---------------------------------------------------------------------------
# _agenerate override: 401 retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agenerate_retries_on_auth_error() -> None:
    """_agenerate should refresh token and retry on AuthenticationError."""
    llm = _make_llm()
    call_count = [0]

    async def mock_parent_agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ):
        call_count[0] += 1
        if call_count[0] == 1:
            raise _make_auth_error()
        return MagicMock(spec=ChatResult)

    with (
        patch.object(
            llm, "_arefresh_copilot_token", new_callable=AsyncMock, return_value=True
        ) as mock_refresh,
        patch.object(llm, "_amaybe_refresh_token_proactively", new_callable=AsyncMock),
        patch("langchain_openai.ChatOpenAI._agenerate", mock_parent_agenerate),
    ):
        await llm._agenerate([HumanMessage("hi")])

    mock_refresh.assert_called_once()
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_agenerate_raises_if_refresh_fails() -> None:
    """_agenerate should re-raise AuthenticationError if refresh returns False."""
    llm = _make_llm()

    async def mock_parent_agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ):
        raise _make_auth_error()

    with (
        patch.object(
            llm, "_arefresh_copilot_token", new_callable=AsyncMock, return_value=False
        ),
        patch.object(llm, "_amaybe_refresh_token_proactively", new_callable=AsyncMock),
        patch("langchain_openai.ChatOpenAI._agenerate", mock_parent_agenerate),
    ):
        with pytest.raises(openai.AuthenticationError):
            await llm._agenerate([HumanMessage("hi")])


# ---------------------------------------------------------------------------
# _astream override: 401 retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_retries_on_auth_error() -> None:
    """_astream should refresh token and retry on AuthenticationError."""
    llm = _make_llm()
    call_count = [0]

    async def mock_parent_astream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> AsyncIterator:
        call_count[0] += 1
        if call_count[0] == 1:
            raise _make_auth_error()
        yield _make_chunk()

    with (
        patch.object(
            llm, "_arefresh_copilot_token", new_callable=AsyncMock, return_value=True
        ) as mock_refresh,
        patch.object(llm, "_amaybe_refresh_token_proactively", new_callable=AsyncMock),
        patch("langchain_openai.ChatOpenAI._astream", mock_parent_astream),
    ):
        chunks = [c async for c in llm._astream([HumanMessage("hi")])]

    mock_refresh.assert_called_once()
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_astream_raises_if_refresh_fails() -> None:
    """_astream should re-raise AuthenticationError if refresh returns False."""
    llm = _make_llm()

    async def mock_parent_astream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> AsyncIterator:
        raise _make_auth_error()
        yield  # make it an async generator

    with (
        patch.object(
            llm, "_arefresh_copilot_token", new_callable=AsyncMock, return_value=False
        ),
        patch.object(llm, "_amaybe_refresh_token_proactively", new_callable=AsyncMock),
        patch("langchain_openai.ChatOpenAI._astream", mock_parent_astream),
    ):
        with pytest.raises(openai.AuthenticationError):
            _ = [c async for c in llm._astream([HumanMessage("hi")])]


# ---------------------------------------------------------------------------
# get_available_models
# ---------------------------------------------------------------------------


def test_get_available_models_uses_cache_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_available_models should use cached copilot token when available."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    future_exp = time.time() + 3600
    cache = {"copilot_token": "cp_cached", "expires_at": future_exp}
    fake_models = [
        {
            "id": "openai/gpt-4.1",
            "supported_endpoints": ["/chat/completions"],
        }
    ]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": fake_models}

    import httpx

    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value=cache,
        ),
        patch.object(
            httpx.Client,
            "__enter__",
            return_value=MagicMock(get=MagicMock(return_value=mock_resp)),
        ),
        patch.object(httpx.Client, "__exit__", return_value=False),
    ):
        models = ChatGithubCopilot.get_available_models()

    assert len(models) == 1
    assert models[0]["id"] == "openai/gpt-4.1"


def test_get_available_models_raises_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_available_models should raise ValueError when no token is available."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value={},
    ):
        with pytest.raises(
            ValueError, match="GitHub token or Copilot token is required"
        ):
            ChatGithubCopilot.get_available_models()


# ---------------------------------------------------------------------------
# Token format helpers
# ---------------------------------------------------------------------------


def test_is_exchangeable_github_token() -> None:
    """Known exchangeable prefixes should return True; others False."""
    assert _is_exchangeable_github_token("gho_abc") is True
    assert _is_exchangeable_github_token("ghp_abc") is True
    assert _is_exchangeable_github_token("ghu_abc") is True
    assert _is_exchangeable_github_token("github_pat_abc") is True
    # Copilot tokens and unknown formats should not be re-exchanged
    assert _is_exchangeable_github_token("tid=abc;exp=1") is False
    assert _is_exchangeable_github_token("some_random_token") is False


def test_is_auth_error_authentication_error() -> None:
    resp = MagicMock()
    resp.status_code = 401
    resp.headers = {}
    exc = openai.AuthenticationError("unauthorized", response=resp, body={})
    assert _is_auth_error(exc) is True


def test_is_auth_error_bad_request_auth_header() -> None:
    assert _is_auth_error(_make_bad_request_auth_error()) is True


def test_is_auth_error_bad_request_non_auth() -> None:
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {}
    exc = openai.BadRequestError("model not found", response=resp, body={})
    assert _is_auth_error(exc) is False


def test_github_pat_token_is_exchanged_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fine-grained PATs (github_pat_xxx) should be exchanged for Copilot tokens."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with (
        patch(
            "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
            return_value={},
        ),
        patch(
            "langchain_githubcopilot_chat.chat_models.fetch_copilot_token",
            return_value=("tid_new_tok", None),
        ) as mock_exchange,
        patch("langchain_githubcopilot_chat.chat_models.save_tokens_to_cache"),
    ):
        llm = ChatGithubCopilot(
            model="openai/gpt-4.1", github_token="github_pat_ABCDEF"
        )

    mock_exchange.assert_called_once_with("github_pat_ABCDEF")
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "tid_new_tok"


# ---------------------------------------------------------------------------
# threading.Lock for sync token refresh
# ---------------------------------------------------------------------------


def test_sync_token_refresh_lock_is_threading_lock() -> None:
    """_sync_token_refresh_lock in auth must be a real threading.Lock."""
    assert isinstance(_auth_module._sync_token_refresh_lock, type(threading.Lock()))
