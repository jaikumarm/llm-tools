from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock

import pytest

from llm_tools import common, scheduler, usage
from llm_tools.capacity import validate_scope
from llm_tools.providers import antigravity
from .conftest import run_cmd


REPORT = (
    "Gemini Models\tWeekly Limit Remaining\t35%\t2030-01-07T00:00:00Z\n"
    "Gemini Models\tFive Hour Limit Remaining\t0%\t2030-01-01T05:00:00Z\n"
    "Claude and GPT models\tWeekly Limit Remaining\t100%\t2030-01-07T00:00:00Z\n"
    "Claude and GPT models\tFive Hour Limit Remaining\t80.5%\t2030-01-01T05:00:00Z\n"
)


def test_report_and_service_round_trip(monkeypatch):
    snap = antigravity.read({"LLM_USAGE_ANTIGRAVITY_USAGE_TEXT": REPORT})
    assert snap.available
    assert [s.remaining_percent for s in snap.scopes] == [35, 0]
    assert not snap.model_scopes
    assert snap.scopes[0].reset_epoch == common.parse_epoch("2030-01-07T00:00:00Z")
    cfg = usage.Config()
    payload = usage.service_payload_from_provider_data(cfg, {"antigravity": snap})
    restored = usage.provider_data_from_service_payload(payload)["antigravity"]
    assert restored == snap
    log_sample = Mock()
    monkeypatch.setattr(common, "log_usage_sample", log_sample)
    usage.log_samples_from_provider_data({"antigravity": restored})
    assert log_sample.call_count == 2
    assert log_sample.call_args.args == ("Antigravity Gemini", "5h", 0)
    rows = usage.antigravity_rows(cfg, restored)
    assert len(rows) == 2
    assert {r.model for r in rows} == {"Gemini"}
    assert len(usage.antigravity_rows(cfg, None)) == 1


@pytest.mark.parametrize("text", [
    "", "Please sign in", "null", "Gemini Models\tWeekly Limit Remaining\tnan%\tbad",
    "Gemini Models\tWeekly Limit Remaining\t101%\tbad",
    "Gemini Models\tWeekly Limit Remaining\t-1%\tbad",
    "Gemini Models\tWeekly Limit Remaining\tnope%\tbad",
    "Gemini Models\tWeekly Limit Remaining\t30\tbad",
    "Gemini Models\tOther Limit\t30%\tbad",
    "Unknown Models\tWeekly Limit Remaining\t30%\tbad",
])
def test_invalid_report_is_unavailable(text):
    snap = antigravity.read({"LLM_USAGE_ANTIGRAVITY_USAGE_TEXT": text})
    assert not snap.available
    assert snap.reason == "inconclusive-usage"


def test_native_command(monkeypatch):
    monkeypatch.setattr(antigravity.shutil, "which", lambda *a, **kw: "/bin/agy")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, REPORT, "ignored"))
    monkeypatch.setattr(antigravity.subprocess, "run", run)
    env = {"LLM_USAGE_ANTIGRAVITY_TIMEOUT": "8"}
    assert antigravity.read(env).available
    assert run.call_args.args[0] == ["/bin/agy", "--print", "/usage", "--print-timeout", "8s"]
    assert run.call_args.kwargs["timeout"] == 10
    assert run.call_args.kwargs["env"] == env
    run.return_value = subprocess.CompletedProcess([], 1, REPORT, "private auth error")
    snap = antigravity.read(env)
    assert not snap.available
    assert snap.reason == "quota-error"
    assert "private" not in repr(snap)


@pytest.mark.parametrize("error, reason", [
    (subprocess.TimeoutExpired("agy", 1), "timeout"), (OSError("failure"), "reader-error"),
])
def test_command_failure(monkeypatch, error, reason):
    monkeypatch.setattr(antigravity.shutil, "which", lambda *a, **kw: "/bin/agy")
    monkeypatch.setattr(antigravity.subprocess, "run", Mock(side_effect=error))
    assert antigravity.read({"LLM_USAGE_ANTIGRAVITY_TIMEOUT": "invalid"}).reason == reason


def test_missing_cli_and_default_environment(monkeypatch):
    monkeypatch.setattr(antigravity.shutil, "which", lambda *a, **kw: None)
    assert antigravity.read({}).reason == "missing-cli"
    monkeypatch.setenv("LLM_USAGE_ANTIGRAVITY_USAGE_TEXT", REPORT)
    assert antigravity.read().available


def test_gemini_gating_ignores_other_families():
    env = {"LLM_USAGE_ANTIGRAVITY_USAGE_TEXT": REPORT, "LLM_USAGE_NOW_EPOCH": "1893456000"}
    snap = common.usage_snapshot_for_provider("antigravity", env)
    decision = common.usage_decision_for_provider("antigravity", "auto", "5", "30", snap, env)
    assert decision["usable"] is False
    assert {w["remaining"] for w in decision["windows"]} == {0, 35}
    report = REPORT.replace("\t0%\t", "\t75%\t").replace("\t80.5%\t", "\t0%\t")
    env["LLM_USAGE_ANTIGRAVITY_USAGE_TEXT"] = report
    snap = common.usage_snapshot_for_provider("antigravity", env)
    assert common.usage_decision_for_provider("antigravity", "auto", "5", "30", snap, env)["usable"]


@pytest.mark.parametrize("attached", [False, True])
def test_launch_command(attached):
    cfg = scheduler.SchedulerConfig(provider="antigravity", model="gemini-4-argon-high", attached=attached)
    assert scheduler.provider_default_argv(cfg, "hello world") == [
        "agy", "--model", cfg.model, "--prompt-interactive" if attached else "--print", "hello world",
    ]
    assert "gemini" in scheduler.scheduler_model_description(cfg)
    cfg.model = None
    assert "Antigravity" in scheduler.scheduler_model_description(cfg)


def test_cli_integration(env):
    env.update(LLM_USAGE_PROVIDERS="antigravity", LLM_USAGE_ANTIGRAVITY_USAGE_TEXT=REPORT)
    result = run_cmd(["./llm-usage", "--json"], env)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["antigravity"]["available"]
    assert len(data["antigravity"]["scopes"]) == 2
    result = run_cmd(["./llm-usage", "--show-source", "--show-remaining-time"], env)
    assert result.returncode == 0, result.stderr
    assert "Antigravity" in result.stdout
    assert "Gemini" in result.stdout
    assert "Claude and GPT models" not in result.stdout
    assert "35%" in result.stdout
    result = run_cmd([
        "./llm-scheduler", "--provider", "antigravity", "--prompt", "hello", "--dry-run",
        "--model", "gemini-4-argon-high", "--command-template", "true",
    ], env)
    assert result.returncode == 0, result.stderr
    result = run_cmd([
        "./ralph-robin", "--providers", "antigravity", "--prompt", "hello", "--dry-run",
        "--command-template", "true", "--max-iterations", "1", "--no-retry",
    ], env)
    assert result.returncode == 0, result.stderr
    validate_scope("antigravity", "weekly")
    with pytest.raises(ValueError):
        validate_scope("antigravity", "balance")
