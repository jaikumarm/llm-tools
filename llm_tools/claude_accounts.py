"""Config directory and provider identity for Claude Code logins."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

ACCOUNT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")


def is_account(provider: str) -> bool:
    return provider.startswith("claude-") and ACCOUNT_NAME.fullmatch(provider[7:]) is not None


def provider_family(provider: str) -> str:
    return "claude" if is_account(provider) else provider


def resolve_directory(value: str, env: Mapping[str, str]) -> Path:
    if value == "~" or value.startswith("~/"):
        value = str(Path(env.get("HOME") or Path.home()) / value[2:])
    return Path(value).expanduser().resolve()


def config_directory(env: Mapping[str, str]) -> Path:
    value = env.get("CLAUDE_CONFIG_DIR")
    if value:
        return resolve_directory(value, env)
    return Path(env.get("HOME") or Path.home()) / ".claude"


def cache_suffix(env: Mapping[str, str]) -> str:
    if not env.get("CLAUDE_CONFIG_DIR"):
        return ""
    directory = config_directory(env)
    if directory == (Path(env.get("HOME") or Path.home()) / ".claude").resolve():
        return ""
    import hashlib

    return "-" + hashlib.sha256(os.fsencode(directory)).hexdigest()[:24]


# These inherited settings can replace the login selected by a named directory.
AUTH_OVERRIDES = (
    "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
