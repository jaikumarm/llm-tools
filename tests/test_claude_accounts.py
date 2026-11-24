from __future__ import annotations

import io
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.error import HTTPError

import pytest

from llm_tools import bridge, bridge_doctor, bridge_sessions, common, config, ralph_robin, routes, scheduler, usage
from llm_tools.capacity import CapacityScope


def credentials(directory: Path, token: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".credentials.json"
    path.write_text(json.dumps({"claudeAiOauth": {"accessToken": token, "refreshToken": f"refresh-{token}"}}))
    return path


def sample(used: int) -> io.BytesIO:
    return io.BytesIO(json.dumps({
        "five_hour": {"utilization": used, "resets_at": "2099-01-01T00:00:00Z"},
        "seven_day": {"utilization": used, "resets_at": "2099-01-02T00:00:00Z"},
    }).encode())


@pytest.fixture
def accounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    default = Path.home() / "personal"
    work = Path.home() / "work account's files"
    path = tmp_path / "config.toml"
    path.write_text('[claude.accounts]\nwork = "~/work account\'s files"\n[usage]\nproviders = ["claude", "claude-work"]\n')
    monkeypatch.setenv("LLM_TOOLS_CONFIG", str(path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/personal")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("LLM_USAGE_LIVE_FETCH_RETRIES", "0")
    monkeypatch.setenv("LLM_USAGE_NOW_EPOCH", "1767225600")
    return default, work


def test_default_override_and_named_account_render_and_sample(accounts, monkeypatch, capsys):
    default, work = accounts
    credentials(Path.home() / ".claude", "wrong")
    credentials(default, "personal")
    credentials(work, "work")
    monkeypatch.setattr(common, "urlopen", lambda req, **_: sample({"Bearer personal": 10, "Bearer work": 70}[req.get_header("Authorization")]))
    cfg = usage.Config()
    data = usage.read_all_provider_data(cfg)
    usage.render_from_provider_data(cfg, data)
    table = capsys.readouterr().out
    assert re.search(r"^Claude +\S+ +5h\s+90%", table, re.MULTILINE)
    assert re.search(r"^claude-work +\S+ +5h\s+30%", table, re.MULTILINE)
    obj = usage.json_object_from_provider_data(cfg, data)
    assert obj["claude"]["five_hour"]["remaining"] == 90
    assert obj["claude-work"]["provider"] == "claude-work"
    assert obj["claude-work"]["five_hour"]["remaining"] == 30
    payload = usage.service_payload_from_provider_data(cfg, data)
    assert usage.provider_data_from_service_payload(payload) == data
    usage.log_samples_from_provider_data(data)
    samples = [json.loads(line) for line in (common.usage_cache_dir() / "llm-usage.log").read_text().splitlines()]
    assert {(item["provider"], item["remaining"]) for item in samples} == {("Claude", 90), ("claude-work", 30)}
    assert bridge_doctor.signin_state("claude")[0] is True
    assert bridge_doctor.signin_state("claude-work")[0] is True


def test_account_cache_and_429_cooldown_are_isolated(accounts, monkeypatch):
    default, work = accounts
    credentials(default, "personal")
    credentials(work, "work")
    calls = []

    def fetch(req, **_):
        token = req.get_header("Authorization")
        calls.append(token)
        if token == "Bearer personal":
            raise HTTPError(req.full_url, 429, "rate limited", {"Retry-After": "120"}, None)
        return sample(35)

    monkeypatch.setattr(common, "urlopen", fetch)
    with common.shared_usage_reads(60):
        assert common.usage_snapshot_for_provider("claude")["reason"] == "rate-limited"
        assert common.usage_snapshot_for_provider("claude-work")["five_hour"]["used"] == 35
    assert common.usage_snapshot_for_provider("claude")["reason"] == "rate-limited"
    assert common.usage_snapshot_for_provider("claude-work")["five_hour"]["used"] == 35
    assert calls == ["Bearer personal", "Bearer work"]
    work_env = common.claude_account_environment("claude-work", os.environ)
    assert common._claude_oauth_rate_limit_path(dict(os.environ)).exists()
    assert not common._claude_oauth_rate_limit_path(work_env).exists()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(work))
    assert common.usage_snapshot_for_provider("claude")["five_hour"]["used"] == 35
    assert len(calls) == 2


def test_refresh_writes_only_to_the_named_account(accounts, monkeypatch):
    default, work = accounts
    personal = credentials(default, "personal")
    personal_before = personal.read_bytes()
    other = credentials(work, "work")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "default-env-token")

    def fetch(req, **_):
        if req.full_url == common.CLAUDE_OAUTH_TOKEN_URL:
            assert json.loads(req.data)["refresh_token"] == "refresh-work"
            return io.BytesIO(b'{"access_token":"new-work","refresh_token":"new-refresh"}')
        token = req.get_header("Authorization")
        if token == "Bearer work":
            raise HTTPError(req.full_url, 401, "expired", {}, None)
        assert token == "Bearer new-work"
        return sample(15)

    monkeypatch.setattr(common, "urlopen", fetch)
    assert common.usage_snapshot_for_provider("claude-work")["five_hour"]["used"] == 15
    assert json.loads(other.read_text())["claudeAiOauth"] == {"accessToken": "new-work", "refreshToken": "new-refresh"}
    assert personal.read_bytes() == personal_before


def test_hidden_account_has_a_json_key_and_is_never_read(accounts, monkeypatch, capsys):
    monkeypatch.setenv("LLM_USAGE_PROVIDERS", "claude")
    monkeypatch.setattr(common, "claude_account_environment", lambda *_: pytest.fail("the hidden account was read"))
    cfg = usage.Config()
    data = usage.read_all_provider_data(cfg)
    usage.render_from_provider_data(cfg, data)
    assert "claude-work" not in capsys.readouterr().out
    assert usage.json_object_from_provider_data(cfg, data)["claude-work"] == {
        "provider": "claude-work", "available": False, "reason": "hidden",
    }
    with pytest.raises(SystemExit, match="2"):
        scheduler.validate_args(scheduler.SchedulerConfig(provider="claude-work", prompt_text="test"))


def test_scheduler_and_ralph_launch_the_named_directory(accounts, tmp_path, monkeypatch):
    _, work = accounts
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "default-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "default-key")
    monkeypatch.setenv("LLM_SCHEDULER_USAGE_JSON", json.dumps({
        "claude": {"five_hour": {"remaining": 0}},
        "claude-work": {"five_hour": {"remaining": 90}},
    }))
    binary = tmp_path / "claude"
    record = tmp_path / "launch.json"
    binary.write_text(
        "#!/bin/sh\n"
        'test -z "${CLAUDE_CODE_OAUTH_TOKEN-}${ANTHROPIC_API_KEY-}" || exit 1\n'
        'python -c \'import json, os, sys; print(json.dumps([os.environ["CLAUDE_CONFIG_DIR"], sys.argv[1:], sys.stdin.read()]))\' "$@"'
        f' > "{record}"\n'
    )
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    assert scheduler.main([
        "--provider", "claude-work", "--scope", "5h", "--prompt", "test",
        "--model", "sonnet", "--headless", "--retry-delays", "", "--log-dir", str(tmp_path / "scheduler"),
    ]) == 0
    assert json.loads(record.read_text()) == [str(work), ["--print", "--model", "sonnet"], "test"]

    with config.config_path().open("a") as stream:
        stream.write('[routes.work-sonnet]\nprovider = "claude-work"\nmodel = "sonnet"\n')
    for rotation in (["--providers", "claude,claude-work"], ["--routes", "claude,work-sonnet"]):
        cfg = ralph_robin.parse_args([*rotation, "--prompt", "test", "--scope", "5h", "--no-even-burn"])
        ralph_robin.apply_config(cfg)
        ralph_robin.validate_args(cfg)
        logs = common.setup_run_logs(tmp_path / "ralph", "test", "claude")
        selection = ralph_robin.select_provider(cfg, logs, 0, set())
        assert selection["provider"] == "claude-work"
        child = ralph_robin.scheduler_config_for(cfg, "claude-work", logs, "test", 1, route_id=selection.get("route", ""))
        assert scheduler.provider_env(child)["CLAUDE_CONFIG_DIR"] == str(work)


def test_bridge_native_account_uses_its_own_launch_and_session_files(accounts, tmp_path, monkeypatch):
    default, work = accounts
    conf = config.load_config()
    assert bridge.creation_rotation("claude,claude-work", None) == ("claude,claude-work", "")
    assert routes.route_or_implicit_provider(conf, "claude-work").provider == "claude-work"
    sid = "01bb6b0d-7d9d-4c10-9d0e-d6eb3c541268"
    project = re.sub(r"[^a-zA-Z0-9]", "-", str(tmp_path.resolve()))
    transcript = work / "projects" / project / f"{sid}.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({"type": "user", "sessionId": sid, "cwd": str(tmp_path)}) + "\n")
    assert bridge_sessions.provider_session_metadata("claude", sid, cwd=tmp_path) is None
    session = bridge_sessions.provider_session_metadata("claude-work", sid, cwd=tmp_path)
    assert session.path == transcript
    assert session.identity.provider == "claude-work"

    logs = common.setup_run_logs(tmp_path / "native", "test", "claude-work")
    cfg = scheduler.SchedulerConfig(
        provider="claude-work", cwd=str(tmp_path), tmux_socket="station.sock", tmux_target="s:w", tmux_pane="%0",
    )
    run = subprocess.run
    probe = tmp_path / "config-dir"
    status = tmp_path / "status"

    def tmux(argv, **_):
        if argv[3] == "display-message":
            return subprocess.CompletedProcess(argv, 0, stdout="1\t\n")
        if argv[3] == "respawn-pane":
            return run(["bash", str(logs.run_dir / "tmux-command.sh")], env={
                **os.environ, "CLAUDE_CONFIG_DIR": str(default), "CLAUDE_CODE_OAUTH_TOKEN": "default-token",
            }, check=True)
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(common, "have_cmd", lambda _: True)
    monkeypatch.setattr(scheduler.subprocess, "run", tmux)
    monkeypatch.setattr(scheduler, "TMUX_EXIT_DRAIN_SECONDS", 0)
    argv = ["sh", "-c", 'test -z "${CLAUDE_CODE_OAUTH_TOKEN-}" && printf %s "$CLAUDE_CONFIG_DIR" > "$1"', "sh", str(probe)]
    assert scheduler.run_tmux(cfg, logs, argv, tmp_path / "out", status) == 0
    assert probe.read_text() == str(work)



def test_no_accounts_preserve_table_and_json_bytes(monkeypatch, capsys):
    monkeypatch.setenv("LLM_USAGE_PROVIDERS", "claude")
    monkeypatch.setenv("LLM_TOOLS_NO_SYMBOLS", "1")
    cfg = usage.Config()
    cfg.no_header = True
    cfg.show_daily_budget = False
    cfg.show_remaining_time = False
    cfg.show_source = False
    cfg.color_enabled = False
    data = {name: usage.unavailable_snapshot(name, "fixture") for name in usage.ALL_PROVIDERS}
    data["codex"] = {"provider": "codex", "available": False}
    data["claude"] = usage.ProviderSnapshot("claude", True, source="fixture", scopes=[
        CapacityScope("5h", usage.CapacityKind.RESET_WINDOW, remaining_percent=75),
        CapacityScope("weekly", usage.CapacityKind.RESET_WINDOW, remaining_percent=50),
    ])
    usage.render_from_provider_data(cfg, data)
    # Captured from the pre-account dashboard with the same fixture.
    assert capsys.readouterr().out == (
        'Claude     yes     5h         75% ████████░░             \n'
        '                   weekly     50% █████░░░░░             \n'
    )
    cfg.json_output = True
    usage.render_from_provider_data(cfg, data, "fixed")
    assert capsys.readouterr().out == (
        '{"generated_at":"fixed","codex":{"provider":"codex","available":false,"reason":"hidden"},'
        '"claude":{"provider":"claude","source":"fixture","five_hour":{"resets_at":null,"used":25.0,"remaining":75.0},'
        '"week":{"resets_at":null,"used":50.0,"remaining":50.0},'
        '"available":true},'
        '"copilot":{"provider":"copilot","available":false,"reason":"hidden"},'
        '"kilo":{"provider":"kilo","available":false,"reason":"hidden"},'
        '"opencode":{"provider":"opencode","available":false,"reason":"hidden"},'
        '"minimax":{"provider":"minimax","available":false,"reason":"hidden"},'
        '"zai":{"provider":"zai","available":false,"reason":"hidden"},'
        '"antigravity":{"provider":"antigravity","available":false,"reason":"hidden"}}\n'
    )
