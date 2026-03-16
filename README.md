# langchain-githubcopilot-chat

This package contains the LangChain integration with GithubcopilotChat

## Installation

```bash
pip install -U langchain-githubcopilot-chat
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatGithubcopilotChat` class exposes chat models from GithubcopilotChat.

```python
from langchain_githubcopilot_chat import ChatGithubcopilotChat

llm = ChatGithubcopilotChat()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`GithubcopilotChatEmbeddings` class exposes embeddings from GithubcopilotChat.

```python
from langchain_githubcopilot_chat import GithubcopilotChatEmbeddings

embeddings = GithubcopilotChatEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs

`GithubcopilotChatLLM` class exposes LLMs from GithubcopilotChat.

```python
from langchain_githubcopilot_chat import GithubcopilotChatLLM

llm = GithubcopilotChatLLM()
llm.invoke("The meaning of life is")
```
