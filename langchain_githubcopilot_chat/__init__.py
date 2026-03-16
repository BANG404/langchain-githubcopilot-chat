from importlib import metadata

from langchain_githubcopilot_chat.chat_models import (
    ChatGithubCopilot,
    ChatGithubcopilotChat,
)
from langchain_githubcopilot_chat.document_loaders import GithubcopilotChatLoader
from langchain_githubcopilot_chat.embeddings import GithubcopilotChatEmbeddings
from langchain_githubcopilot_chat.retrievers import GithubcopilotChatRetriever
from langchain_githubcopilot_chat.toolkits import GithubcopilotChatToolkit
from langchain_githubcopilot_chat.tools import GithubcopilotChatTool
from langchain_githubcopilot_chat.vectorstores import GithubcopilotChatVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatGithubCopilot",
    "ChatGithubcopilotChat",  # backwards-compatible alias
    "GithubcopilotChatVectorStore",
    "GithubcopilotChatEmbeddings",
    "GithubcopilotChatLoader",
    "GithubcopilotChatRetriever",
    "GithubcopilotChatToolkit",
    "GithubcopilotChatTool",
    "__version__",
]
