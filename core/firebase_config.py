"""Load Firebase web config for login / register pages."""

from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "firebase_config.json"


def firebase_config_path() -> Path:
    return _CONFIG_PATH


def load_firebase_config() -> dict[str, str]:
    """
    Read firebase_config.json (or FIREBASE_* env vars).
    Returns {} when not configured yet.
    """
    if _CONFIG_PATH.exists():
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("apiKey") and "YOUR_FIREBASE" not in str(
            data.get("apiKey", "")
        ):
            out = {
                "apiKey": str(data.get("apiKey", "")),
                "authDomain": str(data.get("authDomain", "")),
                "projectId": str(data.get("projectId", "")),
                "storageBucket": str(data.get("storageBucket", "")),
                "messagingSenderId": str(data.get("messagingSenderId", "")),
                "appId": str(data.get("appId", "")),
            }
            if data.get("samlProviderId"):
                out["samlProviderId"] = str(data.get("samlProviderId"))
            return out

    env = {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
    }
    if env["apiKey"] and env["projectId"]:
        return env
    return {}


def is_firebase_configured() -> bool:
    cfg = load_firebase_config()
    return bool(cfg.get("apiKey") and cfg.get("projectId") and cfg.get("appId"))
