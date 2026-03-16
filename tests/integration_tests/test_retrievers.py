from typing import Type

from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)

from langchain_githubcopilot_chat.retrievers import GithubcopilotChatRetriever


class TestGithubcopilotChatRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[GithubcopilotChatRetriever]:
        """Get an empty vectorstore for unit tests."""
        return GithubcopilotChatRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "example query"
