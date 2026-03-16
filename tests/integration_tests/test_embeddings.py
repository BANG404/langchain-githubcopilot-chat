"""Test GithubcopilotChat embeddings."""

from typing import Type

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_githubcopilot_chat.embeddings import GithubcopilotChatEmbeddings


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[GithubcopilotChatEmbeddings]:
        return GithubcopilotChatEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
