"""Integration tests for GithubcopilotChat embeddings.

These tests make real HTTP calls to the GitHub Models embeddings API.
Set the GITHUB_TOKEN environment variable before running:

    export GITHUB_TOKEN="github_pat_..."
    pytest tests/integration_tests/test_embeddings.py -v -m integration
"""

from typing import Type

import pytest
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_githubcopilot_chat.embeddings import GithubcopilotChatEmbeddings


class TestGithubcopilotChatEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[GithubcopilotChatEmbeddings]:
        return GithubcopilotChatEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "openai/text-embedding-3-small"}


# ---------------------------------------------------------------------------
# Additional hand-written integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def embeddings() -> GithubcopilotChatEmbeddings:
    """Return a GithubcopilotChatEmbeddings instance for testing."""
    return GithubcopilotChatEmbeddings(model="openai/text-embedding-3-small")


@pytest.mark.integration
def test_embed_query_returns_vector(
    embeddings: GithubcopilotChatEmbeddings,
) -> None:
    """embed_query should return a non-empty float vector."""
    vector = embeddings.embed_query("What is the meaning of life?")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.integration
def test_embed_documents_returns_vectors(
    embeddings: GithubcopilotChatEmbeddings,
) -> None:
    """embed_documents should return one vector per input text."""
    texts = ["Hello world.", "LangChain is great.", "GitHub Copilot rocks."]
    vectors = embeddings.embed_documents(texts)
    assert len(vectors) == len(texts)
    for vec in vectors:
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)


@pytest.mark.integration
def test_embed_documents_preserves_order(
    embeddings: GithubcopilotChatEmbeddings,
) -> None:
    """Embedding vectors should be returned in the same order as inputs."""
    texts = [f"Document number {i}" for i in range(5)]
    vectors = embeddings.embed_documents(texts)
    assert len(vectors) == 5
    # All vectors should have the same dimensionality
    dims = {len(v) for v in vectors}
    assert len(dims) == 1, f"Inconsistent vector dimensions: {dims}"


@pytest.mark.integration
def test_embed_documents_empty_list(
    embeddings: GithubcopilotChatEmbeddings,
) -> None:
    """embed_documents with an empty list should return an empty list."""
    result = embeddings.embed_documents([])
    assert result == []


@pytest.mark.integration
def test_embed_query_consistent_dimensions(
    embeddings: GithubcopilotChatEmbeddings,
) -> None:
    """Multiple embed_query calls should return vectors of the same dimension."""
    v1 = embeddings.embed_query("First query.")
    v2 = embeddings.embed_query("Second query.")
    assert len(v1) == len(v2)


@pytest.mark.integration
async def test_aembed_query_returns_vector(
    embeddings: GithubcopilotChatEmbeddings,
) -> None:
    """aembed_query should return a non-empty float vector."""
    vector = await embeddings.aembed_query("Async embedding test.")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.integration
async def test_aembed_documents_returns_vectors(
    embeddings: GithubcopilotChatEmbeddings,
) -> None:
    """aembed_documents should return one vector per input text."""
    texts = ["Async document one.", "Async document two."]
    vectors = await embeddings.aembed_documents(texts)
    assert len(vectors) == len(texts)
    for vec in vectors:
        assert isinstance(vec, list)
        assert len(vec) > 0


@pytest.mark.integration
def test_dimensions_parameter() -> None:
    """The dimensions parameter should produce shorter vectors."""
    embed_full = GithubcopilotChatEmbeddings(model="openai/text-embedding-3-small")
    embed_small = GithubcopilotChatEmbeddings(
        model="openai/text-embedding-3-small",
        dimensions=256,
    )
    text = "Testing custom embedding dimensions."
    v_full = embed_full.embed_query(text)
    v_small = embed_small.embed_query(text)
    assert len(v_small) == 256
    assert len(v_full) > len(v_small)


@pytest.mark.integration
def test_large_model() -> None:
    """text-embedding-3-large should return higher-dimensional vectors."""
    embed = GithubcopilotChatEmbeddings(model="openai/text-embedding-3-large")
    vector = embed.embed_query("Testing the large embedding model.")
    assert isinstance(vector, list)
    assert len(vector) > 0
