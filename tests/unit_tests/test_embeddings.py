"""Test embedding model integration."""

from typing import Type

from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_githubcopilot_chat.embeddings import GithubcopilotChatEmbeddings


class TestParrotLinkEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[GithubcopilotChatEmbeddings]:
        return GithubcopilotChatEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
