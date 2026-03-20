"""Authentication utilities for GitHub Copilot."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Callable, Dict, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)

CLIENT_ID = "Iv1.b507a08c87ecfe98"
CACHE_PATH = os.path.expanduser("~/.github-copilot-chat.json")

# Shared Copilot headers
COPILOT_EDITOR_VERSION = "vscode/1.104.1"
COPILOT_PLUGIN_VERSION = "copilot-chat/0.26.7"
COPILOT_INTEGRATION_ID = "vscode-chat"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.26.7"

COPILOT_DEFAULT_HEADERS = {
    "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
    "User-Agent": COPILOT_USER_AGENT,
    "Editor-Version": COPILOT_EDITOR_VERSION,
    "Editor-Plugin-Version": COPILOT_PLUGIN_VERSION,
    "editor-version": COPILOT_EDITOR_VERSION,
    "editor-plugin-version": COPILOT_PLUGIN_VERSION,
    "copilot-vision-request": "true",
}

# In-memory lock for token refresh to prevent concurrent refresh attempts
_token_refresh_lock: Optional[asyncio.Lock] = None
_sync_token_refresh_lock: threading.Lock = threading.Lock()


def _get_token_refresh_lock() -> asyncio.Lock:
    """Get or create the async token refresh lock."""
    global _token_refresh_lock
    if _token_refresh_lock is None:
        _token_refresh_lock = asyncio.Lock()
    return _token_refresh_lock


def save_tokens_to_cache(
    github_token: str,
    copilot_token: str,
    expires_at: Optional[float] = None,
) -> None:
    """Save tokens to cache with optional expiration time."""
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(
                {
                    "github_token": github_token,
                    "copilot_token": copilot_token,
                    "expires_at": expires_at,
                },
                f,
                indent=2,
            )
    except OSError as exc:
        logger.warning("Failed to save Copilot token cache to %s: %s", CACHE_PATH, exc)


def load_tokens_from_cache() -> Dict[str, str]:
    """Load tokens from cache, checking expiration if present."""
    try:
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
            # Check if token has expired
            if data.get("expires_at"):
                if time.time() > data["expires_at"]:
                    # Token expired, return empty
                    return {}
            return data
    except FileNotFoundError:
        return {}  # cache doesn't exist yet — silently OK
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning(
            "Failed to load Copilot token cache from %s: %s", CACHE_PATH, exc
        )
        return {}


def fetch_copilot_token(github_token: str) -> Tuple[Optional[str], Optional[float]]:
    """Fetch copilot token and return it with expiration time.

    Returns:
        Tuple of (token, expires_at_timestamp). expires_at is None if not provided.
    """
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        **COPILOT_DEFAULT_HEADERS,
    }
    with httpx.Client() as client:
        res = client.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers=headers,
        )
        if res.status_code == 200:
            data = res.json()
            token = data.get("token")
            # Copilot tokens typically expire in a few hours
            # The API may return 'expires_at' as a Unix timestamp
            expires_at = data.get("expires_at")
            return token, expires_at
    return None, None


async def afetch_copilot_token(
    github_token: str,
) -> Tuple[Optional[str], Optional[float]]:
    """Async fetch copilot token and return it with expiration time.

    Returns:
        Tuple of (token, expires_at_timestamp). expires_at is None if not provided.
    """
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        **COPILOT_DEFAULT_HEADERS,
    }
    async with httpx.AsyncClient() as client:
        res = await client.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers=headers,
        )
        if res.status_code == 200:
            data = res.json()
            token = data.get("token")
            expires_at = data.get("expires_at")
            return token, expires_at
    return None, None


def get_copilot_token(
    client_id: str = CLIENT_ID,
    callback: Optional[Callable[[str], None]] = None,
    return_both: bool = False,
) -> Union[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Authenticate via GitHub Device Flow to get a Copilot Token.
    This function will block and wait for the user to complete the
    authorization in their browser.

    Args:
        client_id: The GitHub OAuth App Client ID to use. Defaults
            to the VS Code Copilot Chat client ID.
        callback: Optional callable that receives status messages instead of
            printing them. If None, messages are printed to stdout.

    Returns:
        The fetched Copilot Token string, or None if authentication failed.
    """

    def _print(msg: str) -> None:
        if callback:
            callback(msg)
        else:
            print(msg)  # noqa: T201

    _print("1. Requesting device code from GitHub...")
    with httpx.Client() as client:
        res = client.post(
            "https://github.com/login/device/code",
            headers={"Accept": "application/json"},
            data={"client_id": client_id, "scope": "read:user"},
        )
        res.raise_for_status()
        data = res.json()

    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    interval = data.get("interval", 5)

    _print("\n==========================================")
    _print(f"Please open your browser to: {verification_uri}")
    _print(f"And enter the authorization code: {user_code}")
    _print("==========================================\n")
    _print(f"Waiting for authorization (checking every {interval} seconds)...")

    access_token = None
    with httpx.Client() as client:
        while True:
            token_res = client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            ).json()

            if "access_token" in token_res:
                access_token = token_res["access_token"]
                _print("\n✅ Authorization successful! Exchanging for Copilot Token...")
                break
            elif token_res.get("error") == "authorization_pending":
                time.sleep(interval)
            else:
                _print(f"\n❌ Authorization failed: {token_res}")
                return None

        # Exchange the standard access token for a Copilot internal token
        copilot_token, expires_at = fetch_copilot_token(access_token)

        if copilot_token:
            save_tokens_to_cache(access_token, copilot_token, expires_at)
            _print("🎉 Successfully acquired Copilot Token!")
            if return_both:
                return access_token, copilot_token
            return copilot_token
        else:
            _print("❌ Failed to acquire Copilot Token!")
            if return_both:
                return access_token, None
            return None
