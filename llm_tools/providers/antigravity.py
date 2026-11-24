"""Read the shared Gemini quota from Antigravity's tab-separated usage report."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time

from .. import common
from ..bridge_providers import REGISTRY
from ..capacity import CapacityKind, CapacityScope, ProviderSnapshot, PROVIDER_ANTIGRAVITY

SOURCE = "agy /usage"
_WINDOWS = {"Weekly Limit Remaining": "weekly", "Five Hour Limit Remaining": "5h"}
_AUTH_ERROR = re.compile(
    r"\b(?:not (?:signed in|logged in|authenticated)|(?:sign|log) in (?:first|to (?:continue|use))"
    r"|please (?:sign|log) in|authentication required|unauthenticated)\b",
    re.IGNORECASE,
)


def parse_usage(text: str) -> list[CapacityScope]:
    scopes = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            continue
        label, window, percent, reset = (field.strip() for field in fields)
        name = _WINDOWS.get(window)
        if not label.lower().startswith("gemini") or not name or not percent.endswith("%"):
            continue
        remaining = common.num(percent[:-1])
        if remaining is None or not 0 <= remaining <= 100:
            continue
        epoch = common.parse_epoch(reset, use_date=False)
        scopes.append(CapacityScope(
            name=name,
            kind=CapacityKind.RESET_WINDOW,
            remaining_percent=remaining,
            reset_epoch=epoch,
            resets_at=reset if epoch is not None else None,
            source=SOURCE,
        ))
    return scopes


def read(env: dict[str, str] | None = None) -> ProviderSnapshot:
    env = os.environ if env is None else env
    snapshot = ProviderSnapshot(provider=PROVIDER_ANTIGRAVITY, available=False, source=SOURCE)
    report = env.get("LLM_USAGE_ANTIGRAVITY_USAGE_TEXT")
    if report is None:
        cli = shutil.which(REGISTRY[PROVIDER_ANTIGRAVITY].executable, path=env.get("PATH"))
        if cli is None:
            snapshot.reason = "missing-cli"
            return snapshot
        report, snapshot.reason = _live_usage(cli, env)
        if report is None:
            return snapshot
    snapshot.scopes = parse_usage(report)
    snapshot.available = bool(snapshot.scopes)
    if not snapshot.available:
        snapshot.reason = "not-authenticated" if _AUTH_ERROR.search(report) else "inconclusive-usage"
    return snapshot


def _live_usage(cli: str, env: dict[str, str]) -> tuple[str | None, str]:
    timeout = common.num(env.get("LLM_USAGE_ANTIGRAVITY_TIMEOUT"))
    timeout = timeout if timeout is not None and math.isfinite(timeout) and timeout > 0 else 20
    argv = [cli, "--print", "/usage", "--print-timeout", f"{timeout:g}s"]
    attempts, delay = common.live_fetch_retry_plan(env)
    reason = "quota-error"
    for attempt in range(attempts):
        if attempt:
            time.sleep(min(delay, common.live_fetch_retry_max_delay(env)))
            delay *= 2
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, errors="replace",
                stdin=subprocess.DEVNULL, timeout=timeout + 2, env=env,
            )
        except subprocess.TimeoutExpired:
            reason = "timeout"
            continue
        except (OSError, OverflowError):
            return None, "reader-error"

        if _AUTH_ERROR.search(result.stderr) or _AUTH_ERROR.search(result.stdout):
            return None, "not-authenticated"
        if result.returncode == 0:
            return result.stdout, ""
        reason = "quota-error"
    return None, reason
