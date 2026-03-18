from importlib import metadata

from langchain_githubcopilot_chat.auth import get_copilot_token
from langchain_githubcopilot_chat.chat_models import (
    ChatGithubCopilot,
    ChatGithubcopilotChat,
)
from langchain_githubcopilot_chat.embeddings import GithubcopilotChatEmbeddings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

get_available_models = ChatGithubCopilot.get_available_models
get_vscode_token = get_copilot_token

__all__ = [
    "ChatGithubCopilot",
    "ChatGithubcopilotChat",  # backwards-compatible alias
    "GithubcopilotChatEmbeddings",
    "get_copilot_token",
    "get_vscode_token",
    "get_available_models",
    "__version__",
]
