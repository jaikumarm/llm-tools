"""Read Antigravity's native tab-separated ``agy --print /usage`` report."""

from __future__ import annotations

import math
import os
import shutil
import subprocess

from .. import common
from ..capacity import CapacityKind, CapacityScope, ProviderSnapshot


def parse_usage(text: str) -> list[CapacityScope]:
    scopes: list[CapacityScope] = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            continue
        label, window, percent, reset = (field.strip() for field in fields)
        name = {"Weekly Limit Remaining": "weekly", "Five Hour Limit Remaining": "5h"}.get(window)
        if not label.lower().startswith("gemini") or not name or not percent.endswith("%"):
            continue
        try:
            remaining = float(percent[:-1])
        except ValueError:
            continue
        if not math.isfinite(remaining) or not 0 <= remaining <= 100:
            continue
        scopes.append(CapacityScope(
            name=name,
            kind=CapacityKind.RESET_WINDOW,
            remaining_percent=remaining,
            reset_epoch=common.parse_epoch(reset),
            resets_at=reset,
            source="agy /usage",
        ))
    return scopes


def read(env: dict[str, str] | None = None) -> ProviderSnapshot:
    if env is None:
        env = os.environ
    snapshot = ProviderSnapshot(provider="antigravity", available=False, source="agy /usage")
    text = env.get("LLM_USAGE_ANTIGRAVITY_USAGE_TEXT")
    if text is None:
        cli = shutil.which("agy", path=env.get("PATH"))
        if not cli:
            snapshot.reason = "missing-cli"
            return snapshot
        timeout = common.num(env.get("LLM_USAGE_ANTIGRAVITY_TIMEOUT"))
        timeout = timeout if timeout is not None and math.isfinite(timeout) and timeout > 0 else 20
        try:
            result = subprocess.run(
                [cli, "--print", "/usage", "--print-timeout", f"{timeout:g}s"],
                capture_output=True, text=True, timeout=timeout + 2, env=env,
            )
        except subprocess.TimeoutExpired:
            snapshot.reason = "timeout"
            return snapshot
        except OSError:
            snapshot.reason = "reader-error"
            return snapshot
        if result.returncode:
            snapshot.reason = "quota-error"
            return snapshot
        text = result.stdout
    snapshot.scopes = parse_usage(text)
    snapshot.available = bool(snapshot.scopes)
    snapshot.reason = "" if snapshot.available else "inconclusive-usage"
    return snapshot
