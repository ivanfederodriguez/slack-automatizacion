import plistlib
from pathlib import Path

import autostart


def prepare_project(tmp_path):
    project_dir = tmp_path / "project"
    (project_dir / ".venv" / "bin").mkdir(parents=True)
    (project_dir / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    return project_dir


def test_install_launch_agents_writes_expected_plists_and_scripts(tmp_path, monkeypatch):
    project_dir = prepare_project(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(autostart.shutil, "which", lambda name: "/opt/homebrew/bin/ollama")

    launchctl_calls = []

    def fake_run(cmd, check):
        launchctl_calls.append((cmd, check))
        return None

    monkeypatch.setattr(autostart.subprocess, "run", fake_run)
    monkeypatch.setattr(autostart.os, "getuid", lambda: 501)

    artifacts = autostart.install_launch_agents(
        project_dir,
        ollama_base_url="http://127.0.0.1:11434",
    )

    ollama_plist = plistlib.loads(artifacts.ollama_plist.read_bytes())
    agent_plist = plistlib.loads(artifacts.agent_plist.read_bytes())
    sync_plist = plistlib.loads(artifacts.sync_plist.read_bytes())

    assert ollama_plist["KeepAlive"] is True
    assert ollama_plist["ProgramArguments"] == ["/bin/zsh", str(artifacts.ollama_script)]
    assert ollama_plist["StandardOutPath"] == str(artifacts.log_dir / "ollama.stdout.log")
    assert ollama_plist["StandardErrorPath"] == str(artifacts.log_dir / "ollama.stderr.log")

    assert agent_plist["ProgramArguments"] == ["/bin/zsh", str(artifacts.agent_script)]
    assert agent_plist["StandardOutPath"] == str(artifacts.log_dir / "agent.stdout.log")
    assert agent_plist["StandardErrorPath"] == str(artifacts.log_dir / "agent.stderr.log")

    assert sync_plist["KeepAlive"] is True
    assert sync_plist["ProgramArguments"] == ["/bin/zsh", str(artifacts.sync_script)]
    assert sync_plist["StandardOutPath"] == str(artifacts.log_dir / "sync.stdout.log")
    assert sync_plist["StandardErrorPath"] == str(artifacts.log_dir / "sync.stderr.log")

    assert launchctl_calls == [
        (["launchctl", "bootout", "gui/501", str(artifacts.sync_plist)], False),
        (["launchctl", "bootout", "gui/501", str(artifacts.agent_plist)], False),
        (["launchctl", "bootout", "gui/501", str(artifacts.ollama_plist)], False),
        (["launchctl", "bootstrap", "gui/501", str(artifacts.ollama_plist)], True),
        (["launchctl", "bootstrap", "gui/501", str(artifacts.agent_plist)], True),
        (["launchctl", "bootstrap", "gui/501", str(artifacts.sync_plist)], True),
    ]


def test_run_ollama_script_executes_ollama_serve(tmp_path):
    project_dir = tmp_path / "project"

    script = autostart.build_ollama_script(project_dir, "/opt/homebrew/bin/ollama")

    assert "exec '/opt/homebrew/bin/ollama' serve" in script


def test_run_agent_script_waits_for_ollama_before_poll(tmp_path):
    project_dir = prepare_project(tmp_path)

    script = autostart.build_agent_script(
        project_dir,
        python_path=str(project_dir / ".venv" / "bin" / "python"),
        ollama_base_url="http://127.0.0.1:11434",
    )

    assert "until curl -sf 'http://127.0.0.1:11434'/api/tags >/dev/null 2>&1; do" in script
    assert "sleep 2" in script
    assert (
        f"exec '{project_dir / '.venv' / 'bin' / 'python'}' '{project_dir / 'main.py'}' poll" in script
    )


def test_run_sync_worker_executes_trello_done_sync_and_telegram_poll(tmp_path):
    project_dir = prepare_project(tmp_path)

    script = autostart.build_sync_worker_script(
        project_dir,
        python_path=str(project_dir / ".venv" / "bin" / "python"),
        sync_worker_seconds=60,
        sync_waiting_enabled=True,
        sync_trello_done_enabled=True,
        sync_telegram_poll_enabled=True,
        final_reply_mode="telegram_approval",
    )

    assert "while true; do" in script
    assert (
        f"'{project_dir / '.venv' / 'bin' / 'python'}' '{project_dir / 'main.py'}' trello-reply-sync --limit 50"
        in script
    )
    assert (
        f"'{project_dir / '.venv' / 'bin' / 'python'}' '{project_dir / 'main.py'}' trello-waiting-sync --limit 50"
        in script
    )
    assert (
        f"'{project_dir / '.venv' / 'bin' / 'python'}' '{project_dir / 'main.py'}' trello-done-sync --limit 50"
        in script
    )
    assert (
        f"'{project_dir / '.venv' / 'bin' / 'python'}' '{project_dir / 'main.py'}' telegram-poll --limit 20"
        in script
    )
    assert script.index("trello-reply-sync --limit 50") < script.index("trello-waiting-sync --limit 50")
    assert script.index("trello-waiting-sync --limit 50") < script.index("trello-done-sync --limit 50")
    assert '"$FINAL_REPLY_MODE" == "telegram_approval"' in script
    assert 'sleep "$SYNC_WORKER_SECONDS"' in script
