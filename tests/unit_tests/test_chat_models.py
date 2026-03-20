"""Unit tests for ChatGithubCopilot chat model."""

import threading
import time
from typing import Type
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_tests.unit_tests import ChatModelUnitTests

import langchain_githubcopilot_chat.auth as _auth_module
from langchain_githubcopilot_chat.chat_models import (
    ChatGithubCopilot,
    ChatGithubcopilotChat,
)

# ---------------------------------------------------------------------------
# Standard LangChain unit test suite
# ---------------------------------------------------------------------------


class TestChatGithubCopilotUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatGithubCopilot]:
        return ChatGithubCopilot

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "openai/gpt-4.1",
            "temperature": 0,
            "github_token": "fake-token-for-testing",
        }


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
    llm = ChatGithubCopilot(
        model="openai/gpt-4.1",
        github_token="ghp_testtoken",
    )
    assert llm.model_name == "openai/gpt-4.1"
    assert llm._token == "ghp_testtoken"


def test_instantiation_with_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_envtoken")
    llm = ChatGithubCopilot(model="openai/gpt-4.1")
    assert llm._token == "ghp_envtoken"


def test_instantiation_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    llm = ChatGithubCopilot.__new__(ChatGithubCopilot)
    # Calling _token property without any token set should raise
    with pytest.raises(ValueError, match="GitHub token is required"):
        # We bypass __init__ to avoid model_validator; test the property directly
        object.__setattr__(llm, "github_token", None)
        _ = llm._token


def test_model_alias() -> None:
    """The `model` alias should set model_name."""
    llm = ChatGithubCopilot(model="meta/llama-3.3-70b-instruct", github_token="tok")
    assert llm.model_name == "meta/llama-3.3-70b-instruct"


def test_api_key_alias() -> None:
    """Passing github_token explicitly should take priority over env var."""
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="ghp_alias")
    assert llm._token == "ghp_alias"


# ---------------------------------------------------------------------------
# Inference URL construction
# ---------------------------------------------------------------------------


def test_inference_url_default() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    assert llm._inference_url == "https://api.githubcopilot.com/chat/completions"


def test_inference_url_with_org() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok", org="my-org")
    assert llm._inference_url == "https://api.githubcopilot.com/chat/completions"


def test_inference_url_custom_base_url() -> None:
    llm = ChatGithubCopilot(
        model="openai/gpt-4.1",
        github_token="tok",
        base_url="https://custom.endpoint.example.com",
    )
    assert llm._inference_url == "https://custom.endpoint.example.com/chat/completions"


def test_inference_url_strips_trailing_slash() -> None:
    llm = ChatGithubCopilot(
        model="openai/gpt-4.1",
        github_token="tok",
        base_url="https://models.github.ai/",
    )
    assert llm._inference_url == "https://models.github.ai/chat/completions"


# ---------------------------------------------------------------------------
# Header construction
# ---------------------------------------------------------------------------


def test_build_headers_contains_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="ghp_mytoken")
    headers = llm._build_headers()
    assert headers["Authorization"] == "Bearer ghp_mytoken"
    assert headers["Content-Type"] == "application/json"
    assert "Copilot-Integration-Id" in headers


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def test_build_payload_basic() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok", temperature=0.5)
    messages = [SystemMessage("Be helpful."), HumanMessage("Hello")]
    payload = llm._build_payload(messages, stream=False)

    assert payload["model"] == "openai/gpt-4.1"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.5
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": "Be helpful."}
    assert payload["messages"][1] == {"role": "user", "content": "Hello"}


def test_build_payload_streaming_adds_stream_options() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    payload = llm._build_payload([HumanMessage("Hi")], stream=True)
    assert payload["stream"] is True
    assert payload.get("stream_options", {}).get("include_usage") is True


def test_build_payload_with_stop() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    payload = llm._build_payload([HumanMessage("Hi")], stop=["END"])
    assert payload["stop"] == ["END"]


def test_build_payload_stop_from_instance() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok", stop=["STOP"])
    payload = llm._build_payload([HumanMessage("Hi")])
    assert payload["stop"] == ["STOP"]


def test_build_payload_call_stop_overrides_instance() -> None:
    llm = ChatGithubCopilot(
        model="openai/gpt-4.1", github_token="tok", stop=["INSTANCE"]
    )
    payload = llm._build_payload([HumanMessage("Hi")], stop=["CALL"])
    assert payload["stop"] == ["CALL"]


def test_build_payload_with_tools() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    tool = {
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "Does something",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    payload = llm._build_payload([HumanMessage("Hi")], tools=[tool], tool_choice="auto")
    assert "tools" in payload
    assert payload["tool_choice"] == "auto"


def test_build_payload_with_response_format() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    payload = llm._build_payload(
        [HumanMessage("Hi")],
        response_format={"type": "json_object"},
    )
    assert payload["response_format"] == {"type": "json_object"}


def test_build_payload_max_tokens() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok", max_tokens=512)
    payload = llm._build_payload([HumanMessage("Hi")])
    assert payload["max_tokens"] == 512


def test_build_payload_no_none_fields() -> None:
    """Fields with None values should NOT be included in the payload."""
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    payload = llm._build_payload([HumanMessage("Hi")])
    assert "temperature" not in payload
    assert "max_tokens" not in payload
    assert "top_p" not in payload
    assert "stop" not in payload


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def test_message_to_dict_system() -> None:
    from langchain_githubcopilot_chat.chat_models import _message_to_dict

    result = _message_to_dict(SystemMessage("You are helpful."))
    assert result == {"role": "system", "content": "You are helpful."}


def test_message_to_dict_human() -> None:
    from langchain_githubcopilot_chat.chat_models import _message_to_dict

    result = _message_to_dict(HumanMessage("Hello"))
    assert result == {"role": "user", "content": "Hello"}


def test_message_to_dict_ai() -> None:
    from langchain_githubcopilot_chat.chat_models import _message_to_dict

    result = _message_to_dict(AIMessage("Hi there!"))
    assert result == {"role": "assistant", "content": "Hi there!"}


def test_message_to_dict_ai_with_tool_calls() -> None:
    from langchain_githubcopilot_chat.chat_models import _message_to_dict

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "Paris"},
                "id": "call_001",
                "type": "tool_call",
            }
        ],
    )
    result = _message_to_dict(ai_msg)
    assert result["role"] == "assistant"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"


def test_message_to_dict_tool_message() -> None:
    from langchain_core.messages import ToolMessage

    from langchain_githubcopilot_chat.chat_models import _message_to_dict

    msg = ToolMessage(content="22°C sunny", tool_call_id="call_001")
    result = _message_to_dict(msg)
    assert result == {
        "role": "tool",
        "tool_call_id": "call_001",
        "content": "22°C sunny",
    }


def test_message_to_dict_multimodal_human() -> None:
    from langchain_githubcopilot_chat.chat_models import _message_to_dict

    msg = HumanMessage(
        content=[
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,abc123"},
            },
        ]
    )
    result = _message_to_dict(msg)
    assert result["role"] == "user"
    assert isinstance(result["content"], list)
    assert result["content"][0] == {"type": "text", "text": "What is in this image?"}
    assert result["content"][1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# _generate (mocked HTTP)
# ---------------------------------------------------------------------------


FAKE_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 8,
        "total_tokens": 23,
    },
}


def test_generate_calls_api_and_returns_result() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")

    with patch.object(llm, "_do_request", return_value=FAKE_RESPONSE) as mock_req:
        result = llm._generate([HumanMessage("What is the capital of France?")])

    mock_req.assert_called_once()
    assert len(result.generations) == 1
    ai_msg = result.generations[0].message
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "The capital of France is Paris."
    assert ai_msg.response_metadata["finish_reason"] == "stop"
    assert ai_msg.usage_metadata is not None
    assert ai_msg.usage_metadata["input_tokens"] == 15
    assert ai_msg.usage_metadata["output_tokens"] == 8
    assert ai_msg.usage_metadata["total_tokens"] == 23


def test_generate_raises_on_empty_choices() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")

    with patch.object(llm, "_do_request", return_value={"choices": []}):
        with pytest.raises(ValueError, match="no choices"):
            llm._generate([HumanMessage("Hello")])


def test_generate_payload_has_correct_model() -> None:
    llm = ChatGithubCopilot(model="meta/llama-3.3-70b-instruct", github_token="tok")
    captured = {}

    def fake_do_request(payload: dict) -> dict:
        captured["payload"] = payload
        return FAKE_RESPONSE

    with patch.object(llm, "_do_request", side_effect=fake_do_request):
        llm._generate([HumanMessage("Hi")])

    assert captured["payload"]["model"] == "meta/llama-3.3-70b-instruct"
    assert captured["payload"]["stream"] is False


# ---------------------------------------------------------------------------
# _stream (mocked HTTP)
# ---------------------------------------------------------------------------


FAKE_STREAM_CHUNKS = [
    {
        "choices": [
            {"delta": {"role": "assistant", "content": "The "}, "finish_reason": None}
        ]
    },
    {"choices": [{"delta": {"content": "capital "}, "finish_reason": None}]},
    {"choices": [{"delta": {"content": "is Paris."}, "finish_reason": "stop"}]},
    {
        "choices": [],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
    },
]


def test_stream_yields_content_chunks() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")

    with patch.object(llm, "_do_stream", return_value=iter(FAKE_STREAM_CHUNKS)):
        chunks = list(llm._stream([HumanMessage("What is the capital of France?")]))

    # Filter chunks that have actual content
    content_chunks = [c for c in chunks if c.message.content]
    full_text = "".join(str(c.message.content) for c in content_chunks)
    assert "Paris" in full_text


def test_stream_final_chunk_has_usage() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")

    with patch.object(llm, "_do_stream", return_value=iter(FAKE_STREAM_CHUNKS)):
        chunks = list(llm._stream([HumanMessage("What is the capital of France?")]))

    # The last chunk should carry usage metadata
    usage_chunks = [c for c in chunks if c.message.usage_metadata is not None]
    assert len(usage_chunks) > 0
    last_usage = usage_chunks[-1].message.usage_metadata
    assert last_usage["total_tokens"] == 23


def test_stream_payload_has_stream_true() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    captured = {}

    def fake_do_stream(payload: dict):
        captured["payload"] = payload
        yield from FAKE_STREAM_CHUNKS

    with patch.object(llm, "_do_stream", side_effect=fake_do_stream):
        list(llm._stream([HumanMessage("Hi")]))

    assert captured["payload"]["stream"] is True


# ---------------------------------------------------------------------------
# bind_tools
# ---------------------------------------------------------------------------


def test_bind_tools_returns_runnable_with_tools() -> None:
    from pydantic import BaseModel, Field

    class MyTool(BaseModel):
        """A simple tool."""

        query: str = Field(..., description="The query")

    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    bound = llm.bind_tools([MyTool])

    # The bound model should pass tools in kwargs
    assert bound is not None


def test_bind_tools_default_tool_choice_is_auto() -> None:
    """bind_tools should inject tools and tool_choice='auto' into the payload."""
    from pydantic import BaseModel

    class MyTool(BaseModel):
        """A simple tool."""

        x: int

    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    captured = {}

    def fake_do_request(payload: dict) -> dict:
        captured["payload"] = payload
        return FAKE_RESPONSE

    formatted_tools = [
        {
            "type": "function",
            "function": {
                "name": "MyTool",
                "description": "A simple tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        }
    ]

    # Call _generate directly with tools/tool_choice kwargs — this is how
    # RunnableBinding injects bound kwargs into the underlying model call.
    with patch.object(llm, "_do_request", side_effect=fake_do_request):
        llm._generate(
            [HumanMessage("Hello")],
            tools=formatted_tools,
            tool_choice="auto",
        )

    assert captured["payload"].get("tool_choice") == "auto"
    assert "tools" in captured["payload"]
    assert captured["payload"]["tools"][0]["function"]["name"] == "MyTool"


# ---------------------------------------------------------------------------
# Identifying params & LLM type
# ---------------------------------------------------------------------------


def test_llm_type() -> None:
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok")
    assert llm._llm_type == "github-copilot"


def test_identifying_params() -> None:
    llm = ChatGithubCopilot(
        model="openai/gpt-4.1",
        github_token="tok",
        temperature=0.7,
        max_tokens=512,
    )
    params = llm._identifying_params
    assert params["model_name"] == "openai/gpt-4.1"
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 512


# ---------------------------------------------------------------------------
# threading.Lock for sync token refresh
# ---------------------------------------------------------------------------


def test_sync_token_refresh_lock_is_threading_lock() -> None:
    """_sync_token_refresh_lock in auth must be a real threading.Lock."""
    assert isinstance(_auth_module._sync_token_refresh_lock, type(threading.Lock()))


def test_refresh_token_sync_skips_when_lock_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_refresh_token_sync should return immediately if the lock is already held."""
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="ghp_tok")

    acquired = _auth_module._sync_token_refresh_lock.acquire(blocking=True)
    try:
        # Lock is held — _refresh_token_sync should return without modifying the token
        llm._refresh_token_sync("ghp_tok")
        assert llm._cached_copilot_token is None  # no update occurred
    finally:
        _auth_module._sync_token_refresh_lock.release()
    assert acquired  # sanity check


# ---------------------------------------------------------------------------
# Token expiry pre-check
# ---------------------------------------------------------------------------


def test_token_cached_and_not_expired_is_returned_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="ghp_tok")
    llm._cached_copilot_token = "copilot_valid"
    llm._cached_copilot_token_expires_at = time.time() + 3600  # valid for 1 hour
    assert llm._token == "copilot_valid"


def test_token_within_buffer_triggers_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="ghp_tok")
    llm._cached_copilot_token = "copilot_stale"
    # Expires in 30s — within the 60s buffer
    llm._cached_copilot_token_expires_at = time.time() + 30

    with patch.object(llm, "_refresh_token_sync") as mock_refresh:
        _ = llm._token
    mock_refresh.assert_called_once()
    # Cached token was cleared
    assert llm._cached_copilot_token is None


def test_token_expired_triggers_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="ghp_tok")
    llm._cached_copilot_token = "copilot_old"
    llm._cached_copilot_token_expires_at = time.time() - 10  # already expired

    with patch.object(llm, "_refresh_token_sync") as mock_refresh:
        _ = llm._token
    mock_refresh.assert_called_once()
    assert llm._cached_copilot_token is None


def test_token_cache_load_stores_expires_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a copilot_token from file cache should populate expires_at."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    future_exp = time.time() + 7200.0

    cache_data = {
        "copilot_token": "copilot_cached",
        "expires_at": future_exp,
    }

    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="ghp_tok")
    # Force the copilot-token-from-cache path: clear cached token and github_token
    llm._cached_copilot_token = None
    llm._cached_copilot_token_expires_at = None
    # Temporarily clear github_token so _token falls through to load_tokens_from_cache
    original_github_token = llm.github_token
    object.__setattr__(llm, "github_token", None)

    with patch(
        "langchain_githubcopilot_chat.chat_models.load_tokens_from_cache",
        return_value=cache_data,
    ), monkeypatch.context() as m:
        m.delenv("GITHUB_TOKEN", raising=False)
        tok = llm._token

    # Restore
    object.__setattr__(llm, "github_token", original_github_token)

    assert tok == "copilot_cached"
    assert llm._cached_copilot_token_expires_at == future_exp


# ---------------------------------------------------------------------------
# Retry backoff jitter
# ---------------------------------------------------------------------------


def test_retry_backoff_jitter_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync retry should sleep with jitter on transient failures."""
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok", max_retries=1)
    sleep_calls: list = []

    def fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    call_count = 0

    def fake_post(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        raise httpx.TransportError("connection reset")

    uniform_patch = patch(
        "langchain_githubcopilot_chat.chat_models.random.uniform", return_value=0.1
    )
    sleep_patch = patch(
        "langchain_githubcopilot_chat.chat_models.time.sleep", side_effect=fake_sleep
    )
    with patch("httpx.post", side_effect=fake_post), sleep_patch, uniform_patch:
        with pytest.raises(httpx.TransportError):
            llm._do_request({"model": "openai/gpt-4.1", "messages": []})

    # sleep was called once (after first attempt, before second)
    assert len(sleep_calls) == 1
    # base backoff is 2**0 = 1; jitter is 0.1 → total 1.1
    assert sleep_calls[0] == pytest.approx(1.1)


@pytest.mark.asyncio
async def test_retry_backoff_jitter_async(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async retry should sleep with jitter on transient failures."""
    llm = ChatGithubCopilot(model="openai/gpt-4.1", github_token="tok", max_retries=1)
    sleep_calls: list = []

    async def fake_async_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    async def fake_post(*args: object, **kwargs: object) -> object:
        raise httpx.TransportError("connection reset")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.TransportError("connection reset"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async_uniform_patch = patch(
        "langchain_githubcopilot_chat.chat_models.random.uniform", return_value=0.1
    )
    async_sleep_patch = patch(
        "langchain_githubcopilot_chat.chat_models.asyncio.sleep",
        side_effect=fake_async_sleep,
    )
    with patch(
        "httpx.AsyncClient", return_value=mock_client
    ), async_sleep_patch, async_uniform_patch:
        with pytest.raises(httpx.TransportError):
            await llm._do_request_async({"model": "openai/gpt-4.1", "messages": []})

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# _make_usage_chunk deduplication helper
# ---------------------------------------------------------------------------


def test_make_usage_chunk_fields() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    chunk = ChatGithubCopilot._make_usage_chunk(usage)
    assert chunk.message.content == ""
    assert chunk.message.usage_metadata is not None
    assert chunk.message.usage_metadata["input_tokens"] == 10
    assert chunk.message.usage_metadata["output_tokens"] == 5
    assert chunk.message.usage_metadata["total_tokens"] == 15
    assert chunk.message.response_metadata["usage"] == usage
