import time
from typing import Optional

import httpx

CLIENT_ID = "Iv1.b507a08c87ecfe98"


def get_copilot_token(client_id: str = CLIENT_ID) -> Optional[str]:
    """
    Authenticate via GitHub Device Flow to get a Copilot Token.
    This function will block and wait for the user to complete the
    authorization in their browser.

    Args:
        client_id: The GitHub OAuth App Client ID to use. Defaults
            to the VS Code Copilot Chat client ID.

    Returns:
        The fetched Copilot Token string, or None if authentication failed.
    """
    print("1. Requesting device code from GitHub...")  # noqa: T201  # noqa: T201
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

    print("\n==========================================")  # noqa: T201
    print(f"Please open your browser to: {verification_uri}")  # noqa: T201
    print(f"And enter the authorization code: {user_code}")  # noqa: T201
    print("==========================================\n")  # noqa: T201
    print(f"Waiting for authorization (checking every {interval} seconds)...")  # noqa: T201

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
                print(  # noqa: T201  # noqa: T201
                    "\n✅ Authorization successful! Exchanging for Copilot Token..."
                )
                break
            elif token_res.get("error") == "authorization_pending":
                time.sleep(interval)
            else:
                print(f"\n❌ Authorization failed: {token_res}")  # noqa: T201  # noqa: T201
                return None

        # Exchange the standard access token for a Copilot internal token
        copilot_res = client.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "Authorization": f"token {access_token}",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.104.1",
                "Editor-Plugin-Version": "copilot-chat/0.26.7",
            },
        )

        if copilot_res.status_code == 200:
            copilot_token = copilot_res.json().get("token")
            print("🎉 Successfully acquired Copilot Token!")  # noqa: T201  # noqa: T201
            return copilot_token
        else:
            print(f"❌ Failed to acquire Copilot Token: {copilot_res.text}")  # noqa: T201  # noqa: T201
            return None
