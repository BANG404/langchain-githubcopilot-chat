"""GitHub Copilot Chat model integration via the OpenAI-compatible API."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
import openai
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, model_validator

from langchain_githubcopilot_chat.auth import (
    COPILOT_DEFAULT_HEADERS,
    _get_token_refresh_lock,
    _sync_token_refresh_lock,
    afetch_copilot_token,
    fetch_copilot_token,
    load_tokens_from_cache,
    save_tokens_to_cache,
)

logger = logging.getLogger(__name__)

_GITHUB_COPILOT_BASE_URL = "https://api.githubcopilot.com"

# Buffer (seconds) before token expiry to trigger a proactive refresh
_TOKEN_REFRESH_BUFFER_SECS: int = 60

# GitHub token prefixes that can be exchanged for a short-lived Copilot token.
# Copilot tokens themselves start with "tid=" and must NOT be re-exchanged.
_EXCHANGEABLE_TOKEN_PREFIXES = ("gho_", "ghp_", "ghu_", "github_pat_")


def _is_exchangeable_github_token(token: str) -> bool:
    """Return True if *token* should be exchanged for a Copilot token."""
    return token.startswith(_EXCHANGEABLE_TOKEN_PREFIXES)


def _is_auth_error(exc: Exception) -> bool:
    """Return True for 401 AuthenticationError OR 400 badly-formatted-auth BadRequestError."""  # noqa: E501
    if isinstance(exc, openai.AuthenticationError):
        return True
    if isinstance(exc, openai.BadRequestError):
        msg = str(exc).lower()
        return "authorization" in msg or "badly formatted" in msg
    return False


class ChatGithubCopilot(ChatOpenAI):
    """GitHub Copilot Chat model via the OpenAI-compatible API.

    Uses ``langchain-openai`` under the hood, pointing at the GitHub Copilot
    inference endpoint.  Handles GitHub token → Copilot token exchange and
    caching automatically.

    Setup:
        Install ``langchain-githubcopilot-chat`` and set the ``GITHUB_TOKEN``
        environment variable (a classic or fine-grained PAT with the
        ``models: read`` scope, or a GitHub Copilot subscription token).

        .. code-block:: bash

            pip install -U langchain-githubcopilot-chat
            export GITHUB_TOKEN="github_pat_..."

    Key init args — completion params:
        model: str
            Model ID in the ``{publisher}/{model_name}`` format, e.g.
            ``"openai/gpt-4.1"`` or ``"meta/llama-3.3-70b-instruct"``.
        temperature: Optional[float]
            Sampling temperature in ``[0, 1]``.
        max_tokens: Optional[int]
            Maximum number of tokens to generate.

    Key init args — client params:
        github_token: Optional[SecretStr]
            GitHub token.  Falls back to the ``GITHUB_TOKEN`` env var.

    Instantiate:
        .. code-block:: python

            from langchain_githubcopilot_chat import ChatGithubCopilot

            llm = ChatGithubCopilot(
                model="openai/gpt-4.1",
                temperature=0,
                max_tokens=1024,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate to French."),
                ("human", "I love programming."),
            ]
            ai_msg = llm.invoke(messages)
            print(ai_msg.content)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            ai_msg = await llm.ainvoke(messages)

            async for chunk in llm.astream(messages):
                print(chunk.content, end="", flush=True)

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location.'''
                location: str = Field(
                    ..., description="City and state, e.g. Paris, France"
                )

            llm_with_tools = llm.bind_tools([GetWeather])
            ai_msg = llm_with_tools.invoke("What is the weather like in Paris?")
            print(ai_msg.tool_calls)
    """

    github_token: Optional[SecretStr] = Field(default=None)
    """GitHub token with ``models: read`` scope.

    If not provided, the value of the ``GITHUB_TOKEN`` environment variable
    is used and automatically exchanged for a short-lived Copilot token.
    """

    @model_validator(mode="before")
    @classmethod
    def _setup_copilot_auth(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve credentials and configure OpenAI-compatible fields.

        Priority order for the GitHub token:
        1. Explicitly passed ``github_token``
        2. ``GITHUB_TOKEN`` environment variable
        3. ``~/.github-copilot-chat.json`` cache file

        If the resolved token is a standard GitHub OAuth/PAT token it is
        exchanged for a short-lived Copilot token (cached to disk).
        """
        # 1. Resolve raw GitHub token
        github_token = values.get("github_token") or os.environ.get("GITHUB_TOKEN")

        if not github_token:
            cached = load_tokens_from_cache()
            github_token = cached.get("github_token")

        # 2. Get/exchange for a Copilot token
        api_token: Optional[str] = None

        if github_token:
            github_token_str = (
                github_token.get_secret_value()
                if hasattr(github_token, "get_secret_value")
                else str(github_token)
            )

            # Always persist the resolved github_token so _refresh_copilot_token
            # can use it even when the original token came from the file cache.
            values["github_token"] = github_token_str

            # Try cached Copilot token first
            cached = load_tokens_from_cache()
            cached_token = cached.get("copilot_token")
            cached_exp = cached.get("expires_at")

            if cached_token and (
                cached_exp is None
                or time.time() < float(cached_exp) - _TOKEN_REFRESH_BUFFER_SECS
            ):
                api_token = cached_token
            elif _is_exchangeable_github_token(github_token_str):
                # Exchange GitHub token for a Copilot token
                new_token, expires_at = fetch_copilot_token(github_token_str)
                if new_token:
                    save_tokens_to_cache(github_token_str, new_token, expires_at)
                    api_token = new_token

            if not api_token:
                # Fall back to using the raw token (e.g. fine-grained PATs,
                # enterprise tokens, or environments without network access).
                api_token = github_token_str

        if not api_token:
            raise ValueError(
                "A GitHub token is required. Set the GITHUB_TOKEN environment "
                "variable, pass ``github_token``, or run ``get_copilot_token()`` "
                "to authenticate."
            )

        # 3. Configure the underlying ChatOpenAI fields
        values["openai_api_key"] = api_token
        values.setdefault("openai_api_base", _GITHUB_COPILOT_BASE_URL)

        # Merge Copilot-required headers with any user-supplied ones
        user_headers: Dict[str, str] = values.get("default_headers") or {}
        values["default_headers"] = {**COPILOT_DEFAULT_HEADERS, **user_headers}

        return values

    @property
    def _llm_type(self) -> str:
        return "github-copilot"

    # ------------------------------------------------------------------
    # Token refresh helpers
    # ------------------------------------------------------------------

    def _get_github_token_str(self) -> str:
        """Return the underlying GitHub OAuth token string."""
        if self.github_token:
            return self.github_token.get_secret_value()
        env = os.environ.get("GITHUB_TOKEN", "")
        if env:
            return env
        cached = load_tokens_from_cache()
        return cached.get("github_token", "")

    def _refresh_copilot_token(self) -> bool:
        """Synchronously fetch a new Copilot token and rebuild the OpenAI clients.

        Returns True if the token was refreshed successfully.
        """
        if not _sync_token_refresh_lock.acquire(blocking=False):
            # Another thread is refreshing; wait for it to finish, then return.
            _sync_token_refresh_lock.acquire()
            _sync_token_refresh_lock.release()
            return False
        try:
            gh_token = self._get_github_token_str()
            if not gh_token or not _is_exchangeable_github_token(gh_token):
                logger.warning(
                    "Cannot refresh Copilot token: no exchangeable GitHub "  # noqa: E501
                    "token available (token prefix: %s...).",
                    gh_token[:8] if gh_token else "<empty>",
                )
                return False

            new_token, expires_at = fetch_copilot_token(gh_token)
            if not new_token:
                logger.warning("Copilot token refresh returned no token.")
                return False

            save_tokens_to_cache(gh_token, new_token, expires_at)
            self.openai_api_key = SecretStr(new_token)
            self._rebuild_clients()
            logger.debug("Copilot token refreshed successfully.")
            return True
        finally:
            _sync_token_refresh_lock.release()

    async def _arefresh_copilot_token(self) -> bool:
        """Asynchronously fetch a new Copilot token and rebuild the OpenAI clients.

        Returns True if the token was refreshed successfully.
        """
        lock = _get_token_refresh_lock()
        async with lock:
            gh_token = self._get_github_token_str()
            if not gh_token or not _is_exchangeable_github_token(gh_token):
                logger.warning(
                    "Cannot refresh Copilot token: no exchangeable GitHub "  # noqa: E501
                    "token available (token prefix: %s...).",
                    gh_token[:8] if gh_token else "<empty>",
                )
                return False

            new_token, expires_at = await afetch_copilot_token(gh_token)
            if not new_token:
                logger.warning("Copilot token refresh returned no token.")
                return False

            save_tokens_to_cache(gh_token, new_token, expires_at)
            self.openai_api_key = SecretStr(new_token)
            self._rebuild_clients()
            logger.debug("Copilot token refreshed successfully (async).")
            return True

    def _rebuild_clients(self) -> None:
        """Nullify and rebuild the underlying OpenAI sync/async clients."""
        self.client = None
        self.async_client = None
        self.root_client = None
        self.root_async_client = None
        # validate_environment re-creates the clients from the current field values.
        self.validate_environment()  # type: ignore[operator]

    def _maybe_refresh_token_proactively(self) -> None:
        """Check the cached token expiry and refresh proactively if needed."""
        cached = load_tokens_from_cache()
        expires_at = cached.get("expires_at")
        if (
            expires_at is not None
            and time.time() >= float(expires_at) - _TOKEN_REFRESH_BUFFER_SECS
        ):
            logger.debug("Copilot token near/past expiry — proactively refreshing.")
            self._refresh_copilot_token()

    async def _amaybe_refresh_token_proactively(self) -> None:
        """Async version: check the cached token expiry and refresh if needed."""
        cached = load_tokens_from_cache()
        expires_at = cached.get("expires_at")
        if (
            expires_at is not None
            and time.time() >= float(expires_at) - _TOKEN_REFRESH_BUFFER_SECS
        ):
            logger.debug(
                "Copilot token near/past expiry — proactively refreshing (async)."
            )
            await self._arefresh_copilot_token()

    # ------------------------------------------------------------------
    # Overrides with token-refresh retry logic
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._maybe_refresh_token_proactively()
        try:
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc):
                raise
            logger.warning("Copilot token rejected; refreshing and retrying. %s", exc)
            if self._refresh_copilot_token():
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._maybe_refresh_token_proactively()
        try:
            yield from super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc):
                raise
            logger.warning("Copilot token rejected; refreshing and retrying. %s", exc)
            if self._refresh_copilot_token():
                yield from super()._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                raise

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        await self._amaybe_refresh_token_proactively()
        try:
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc):
                raise
            logger.warning("Copilot token rejected; refreshing and retrying. %s", exc)
            if await self._arefresh_copilot_token():
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            raise

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        await self._amaybe_refresh_token_proactively()
        try:
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc):
                raise
            logger.warning("Copilot token rejected; refreshing and retrying. %s", exc)
            if await self._arefresh_copilot_token():
                async for chunk in super()._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                ):
                    yield chunk
            else:
                raise

    @classmethod
    def get_available_models(
        cls,
        github_token: Optional[str] = None,
        copilot_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get the list of available models from the GitHub Copilot API.

        Resolution order:
        1. Explicit ``copilot_token`` parameter.
        2. Cached copilot token from ``~/.github-copilot-chat.json``.
        3. Exchange ``github_token`` / ``GITHUB_TOKEN`` env var for a copilot token.
        """
        token = copilot_token

        if not token:
            cached = load_tokens_from_cache()
            token = cached.get("copilot_token")

        if not token:
            gh_token = github_token or os.environ.get("GITHUB_TOKEN")
            if not gh_token:
                raise ValueError(
                    "A GitHub token or Copilot token is required. Set the "
                    "GITHUB_TOKEN environment variable, pass ``github_token``, "
                    "or pass ``copilot_token``."
                )
            if _is_exchangeable_github_token(gh_token):
                exchanged, _ = fetch_copilot_token(gh_token)
                if exchanged:
                    token = exchanged
            if not token:
                token = gh_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            **COPILOT_DEFAULT_HEADERS,
        }

        with httpx.Client() as client:
            response = client.get(f"{_GITHUB_COPILOT_BASE_URL}/models", headers=headers)
            response.raise_for_status()
            all_models: List[Dict[str, Any]] = response.json().get("data", [])

        return [m for m in all_models if _supports_chat_completions(m)]


def _supports_chat_completions(model: Dict[str, Any]) -> bool:
    """Return True if *model* supports the ``/chat/completions`` endpoint.

    Models that omit ``supported_endpoints`` are legacy Azure OpenAI models
    that have always been served via ``/chat/completions``.
    """
    endpoints = model.get("supported_endpoints")
    if endpoints is None:
        # Field absent → legacy model, assume chat/completions
        return True
    return "/chat/completions" in endpoints


# ---------------------------------------------------------------------------
# Backwards-compatible alias (matches the generated stub name)
# ---------------------------------------------------------------------------

ChatGithubcopilotChat = ChatGithubCopilot
