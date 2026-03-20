# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-03-20

### Fixed
- `_do_stream` and `_do_stream_async` now retry on transient network errors
  (`httpx.TimeoutException`, `httpx.TransportError` including `ReadError`) with
  the same exponential back-off + 25 % jitter already used by `_do_request` /
  `_do_request_async`.  Previously the streaming paths had no retry logic,
  causing `httpx.ReadError` (connection dropped before response headers were
  received) to surface as an unhandled exception.
- `_do_stream` and `_do_stream_async` now handle HTTP 401 responses by
  refreshing the GitHub Copilot token and retrying the request, consistent with
  the non-streaming paths.

## [0.5.0] - 2026-03-20

### Added
- `_make_usage_chunk` static method on `ChatGithubCopilot` to eliminate code
  duplication between `_stream` and `_astream` usage-only final chunk handling.
- `_cached_copilot_token_expires_at` private attribute to track Copilot token
  expiry in memory, enabling proactive refresh before tokens expire.
- `_TOKEN_REFRESH_BUFFER_SECS` constant (60 s) controlling how early before
  token expiry a proactive refresh is triggered.
- Logging via `logging.getLogger(__name__)` in both `auth` and `chat_models`
  modules for better observability of token-management operations.

### Changed
- **Thread safety**: replaced the boolean `_sync_token_refresh_lock` flag in
  `auth.py` with a proper `threading.Lock`, eliminating the race condition
  where two threads could concurrently enter `_refresh_token_sync`.
- **Proactive token expiry**: `_token` property now checks
  `_cached_copilot_token_expires_at` and clears stale tokens before they
  expire (within the 60-second buffer), preventing 401 errors on expiry.
- **Retry jitter**: exponential backoff in `_do_request` / `_do_request_async`
  (both `ChatGithubCopilot` and `GithubcopilotChatEmbeddings`) now adds up to
  25 % random jitter to avoid thundering-herd retry storms.
- **Retry sleep for embeddings**: `GithubcopilotChatEmbeddings._do_request` and
  `_do_request_async` now sleep between retries (previously retried immediately
  with no delay).
- **Top-level imports**: `asyncio`, `logging`, `random`, `threading`, and `time`
  moved from function-level late imports to module-level in `chat_models.py`.
- **Exception specificity** in `auth.py`:
  - `save_tokens_to_cache` now catches `OSError` only and logs a warning
    instead of silently swallowing all exceptions.
  - `load_tokens_from_cache` now catches `FileNotFoundError` silently (cache
    absent is expected) and `(OSError, json.JSONDecodeError, KeyError, ValueError)`
    with a warning log, rather than catching all exceptions silently.
- **Exception specificity** in `chat_models.py`: token-exchange failures in the
  `_token` property now catch `(httpx.NetworkError, httpx.TimeoutException,
  OSError)` explicitly and log at DEBUG level.
- `_refresh_token_sync` and `_refresh_token_async` now persist the fetched
  `expires_at` timestamp to `_cached_copilot_token_expires_at`.
- `load_tokens_from_cache` result is now used to populate
  `_cached_copilot_token_expires_at` when a valid cached Copilot token is found.

### Fixed
- Concurrent calls to `_refresh_token_sync` from multiple threads no longer
  race on a shared boolean flag; the `threading.Lock` ensures only one refresh
  proceeds while others skip gracefully.

## [0.4.0] - 2025-05-01

### Added
- Initial public release with `ChatGithubCopilot` and `GithubcopilotChatEmbeddings`.
- Support for streaming, tool calling, structured output, and multimodal image input.
- Device-flow authentication via `get_copilot_token()`.
- Token caching to `~/.github-copilot-chat.json`.
