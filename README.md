# LangChain GitHub Copilot Chat

This package provides a LangChain integration for **GitHub Copilot**, allowing you to use Copilot's models (including GPT-4o, Claude 3.5 Sonnet, etc.) as standard LangChain `BaseChatModel` components.

Unlike other integrations, this package mimics the official VS Code Copilot Chat extension behavior, providing access to the full suite of models available to Copilot subscribers.

## 🚀 Features

- **Real Copilot API**: Connects to `api.githubcopilot.com` using official VS Code headers.
- **Easy Auth**: Built-in GitHub Device Flow for acquiring a valid Copilot Token.
- **Model Discovery**: Dynamic fetching of all models authorized for your account.
- **LangChain Native**: Full support for Streaming, Tool Calling, and Async operations.

## 📦 Installation

```bash
pip install -U langchain-githubcopilot-chat
```

## 🔐 Authentication

To use GitHub Copilot, you need a valid Copilot Token. You can obtain one interactively using the built-in helper:

```python
from langchain_githubcopilot_chat import get_vscode_token

# This will prompt you to visit a GitHub URL and enter a code
token = get_vscode_token()
print(f"Your Token: {token}")
```

For custom output handling (e.g., in GUI applications), pass a callback:

```python
from langchain_githubcopilot_chat import get_copilot_token

def on_message(msg):
    # Handle status messages (e.g., display in UI)
    print(f"[Copilot] {msg}")

token = get_vscode_token(callback=on_message)
```

Alternatively, set it as an environment variable:
```bash
export GITHUB_TOKEN="your_copilot_token_here"
```

## 🛠 Usage

### Chat Models

Access any model supported by Copilot (e.g., `gpt-4o`, `gpt-4o-mini`, `claude-3.5-sonnet`).

```python
from langchain_githubcopilot_chat import ChatGithubCopilot

# Initialize with a specific model
llm = ChatGithubCopilot(
    model="gpt-4o", 
    temperature=0.7
)

# Simple invocation
response = llm.invoke("Explain Quantum Entanglement in one sentence.")
print(response.content)

# Streaming
for chunk in llm.stream("Write a short poem about coding."):
    print(chunk.content, end="", flush=True)
```

### Discovery Available Models

GitHub Copilot periodically updates its available models. You can list what's currently available for your token:

```python
from langchain_githubcopilot_chat import get_available_models

models = get_available_models()
for model in models:
    print(f"ID: {model['id']} - Name: {model.get('name')}")
```

### Embeddings

Use Copilot's embedding models for RAG or semantic search:

```python
from langchain_githubcopilot_chat import GithubcopilotChatEmbeddings

embeddings = GithubcopilotChatEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("GitHub Copilot is awesome!")
```

## 📖 Advanced: Tool Calling

```python
from pydantic import BaseModel, Field
from langchain_githubcopilot_chat import ChatGithubCopilot

class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

llm = ChatGithubCopilot(model="gpt-4o")
llm_with_tools = llm.bind_tools([GetWeather])

ai_msg = llm_with_tools.invoke("What's the weather like in Tokyo?")
print(ai_msg.tool_calls)
```

## ⚖️ Disclaimer

This project is an independent community integration and is not affiliated with, endorsed by, or supported by GitHub, Inc. Usage of this package must comply with GitHub's [Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service).