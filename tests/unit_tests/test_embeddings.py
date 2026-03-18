"""Unit tests for GithubcopilotChatEmbeddings."""

from typing import Type
from unittest.mock import AsyncMock, patch

import pytest
from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_githubcopilot_chat.embeddings import GithubcopilotChatEmbeddings


class TestGithubcopilotChatEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[GithubcopilotChatEmbeddings]:
        return GithubcopilotChatEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model": "openai/text-embedding-3-small",
            "github_token": "fake-token-for-testing",
        }


# ---------------------------------------------------------------------------
# Instantiation & field resolution
# ---------------------------------------------------------------------------


def test_instantiation_with_explicit_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly supplied token should take priority over GITHUB_TOKEN env var."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small",
        github_token="ghp_testtoken",
    )
    assert embed.model_name == "openai/text-embedding-3-small"
    assert embed._token == "ghp_testtoken"


def test_instantiation_with_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_envtoken")
    embed = GithubcopilotChatEmbeddings(model="openai/text-embedding-3-small")
    assert embed._token == "ghp_envtoken"


def test_instantiation_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    embed = GithubcopilotChatEmbeddings.__new__(GithubcopilotChatEmbeddings)
    object.__setattr__(embed, "github_token", None)
    with pytest.raises(ValueError, match="GitHub token is required"):
        _ = embed._token


def test_model_alias() -> None:
    """The `model` alias should set model_name."""
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-large",
        github_token="tok",
    )
    assert embed.model_name == "openai/text-embedding-3-large"


def test_optional_dimensions() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small",
        github_token="tok",
        dimensions=256,
    )
    assert embed.dimensions == 256


# ---------------------------------------------------------------------------
# Endpoint URL construction
# ---------------------------------------------------------------------------


def test_embeddings_url_default() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )
    assert embed._embeddings_url == "https://api.githubcopilot.com/embeddings"


def test_embeddings_url_with_org() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small",
        github_token="tok",
        org="my-org",
    )
    assert embed._embeddings_url == "https://api.githubcopilot.com/embeddings"


def test_embeddings_url_strips_trailing_slash() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small",
        github_token="tok",
        base_url="https://models.github.ai/",
    )
    assert embed._embeddings_url == "https://models.github.ai/embeddings"


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def test_build_headers_contains_auth() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="ghp_mytoken"
    )
    headers = embed._build_headers()
    assert headers["Authorization"] == "Bearer ghp_mytoken"
    assert headers["Content-Type"] == "application/json"
    assert "Copilot-Integration-Id" in headers


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def test_build_payload_single_string() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )
    payload = embed._build_payload("hello world")
    assert payload["model"] == "openai/text-embedding-3-small"
    assert payload["input"] == "hello world"
    assert payload["encoding_format"] == "float"
    assert "dimensions" not in payload


def test_build_payload_list_of_strings() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )
    payload = embed._build_payload(["text one", "text two"])
    assert payload["input"] == ["text one", "text two"]


def test_build_payload_with_dimensions() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small",
        github_token="tok",
        dimensions=512,
    )
    payload = embed._build_payload(["hello"])
    assert payload["dimensions"] == 512


# ---------------------------------------------------------------------------
# _extract_embeddings helper
# ---------------------------------------------------------------------------


def test_extract_embeddings_preserves_order() -> None:
    response = {
        "data": [
            {"index": 1, "embedding": [0.2, 0.3]},
            {"index": 0, "embedding": [0.0, 0.1]},
        ]
    }
    result = GithubcopilotChatEmbeddings._extract_embeddings(response)
    assert result == [[0.0, 0.1], [0.2, 0.3]]


def test_extract_embeddings_raises_on_empty_data() -> None:
    with pytest.raises(ValueError, match="no data"):
        GithubcopilotChatEmbeddings._extract_embeddings({"data": []})


# ---------------------------------------------------------------------------
# embed_documents / embed_query (mocked HTTP)
# ---------------------------------------------------------------------------

FAKE_EMBED_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
        {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
    ],
    "model": "openai/text-embedding-3-small",
    "usage": {"prompt_tokens": 10, "total_tokens": 10},
}

FAKE_SINGLE_EMBED_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.7, 0.8, 0.9]},
    ],
    "model": "openai/text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}


def test_embed_documents_returns_vectors() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )

    with patch.object(embed, "_do_request", return_value=FAKE_EMBED_RESPONSE):
        result = embed.embed_documents(["doc one", "doc two"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]


def test_embed_documents_empty_list() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )
    result = embed.embed_documents([])
    assert result == []


def test_embed_query_returns_single_vector() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )

    with patch.object(embed, "_do_request", return_value=FAKE_SINGLE_EMBED_RESPONSE):
        result = embed.embed_query("hello world")

    assert result == [0.7, 0.8, 0.9]


def test_embed_documents_payload_model_name() -> None:
    """The correct model name should be sent in the request payload."""
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-large", github_token="tok"
    )
    captured: dict = {}

    def fake_do_request(payload: dict) -> dict:
        captured["payload"] = payload
        return FAKE_EMBED_RESPONSE

    with patch.object(embed, "_do_request", side_effect=fake_do_request):
        embed.embed_documents(["test"])

    assert captured["payload"]["model"] == "openai/text-embedding-3-large"
    assert captured["payload"]["input"] == ["test"]


# ---------------------------------------------------------------------------
# Async embed (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aembed_documents_returns_vectors() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )

    with patch.object(
        embed,
        "_do_request_async",
        new=AsyncMock(return_value=FAKE_EMBED_RESPONSE),
    ):
        result = await embed.aembed_documents(["doc one", "doc two"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_aembed_query_returns_single_vector() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )

    with patch.object(
        embed,
        "_do_request_async",
        new=AsyncMock(return_value=FAKE_SINGLE_EMBED_RESPONSE),
    ):
        result = await embed.aembed_query("hello world")

    assert result == [0.7, 0.8, 0.9]


@pytest.mark.asyncio
async def test_aembed_documents_empty_list() -> None:
    embed = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small", github_token="tok"
    )
    result = await embed.aembed_documents([])
    assert result == []
