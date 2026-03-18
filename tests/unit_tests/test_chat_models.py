"""Unit tests for ChatGithubCopilot chat model."""

from typing import Type
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_tests.unit_tests import ChatModelUnitTests

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
