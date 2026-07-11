from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


OLLAMA_LABEL = "com.ivanrodriguez.slack-agent.ollama"
AGENT_LABEL = "com.ivanrodriguez.slack-agent.agent"
SYNC_LABEL = "com.ivanrodriguez.slack-agent.sync"


@dataclass(frozen=True)
class LaunchArtifacts:
    ollama_plist: Path
    agent_plist: Path
    sync_plist: Path
    ollama_script: Path
    agent_script: Path
    sync_script: Path
    log_dir: Path


def ensure_executable(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(current_mode | 0o111)


def build_launch_artifacts(project_dir: Path) -> LaunchArtifacts:
    support_dir = Path.home() / "Library" / "Application Support" / "slack-personal-agent" / "runtime"
    launch_dir = Path.home() / "Library" / "LaunchAgents"
    log_dir = support_dir / "logs"
    support_dir.mkdir(parents=True, exist_ok=True)
    launch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return LaunchArtifacts(
        ollama_plist=launch_dir / f"{OLLAMA_LABEL}.plist",
        agent_plist=launch_dir / f"{AGENT_LABEL}.plist",
        sync_plist=launch_dir / f"{SYNC_LABEL}.plist",
        ollama_script=support_dir / "run_ollama.sh",
        agent_script=support_dir / "run_agent.sh",
        sync_script=support_dir / "run_sync_worker.sh",
        log_dir=log_dir,
    )


def build_ollama_script(project_dir: Path, ollama_path: str) -> str:
    return f"""#!/bin/zsh
set -euo pipefail
cd {shell_quote(str(project_dir))}
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
exec {shell_quote(ollama_path)} serve
"""


def _healthcheck_curl_command(ollama_base_url: str, ollama_auth_token: str = "") -> str:
    auth_header = (
        f" -H {shell_quote(f'Authorization: Bearer {ollama_auth_token}')}"
        if ollama_auth_token
        else ""
    )
    return f"curl -sf{auth_header} {shell_quote(ollama_base_url.rstrip('/'))}/api/tags"


def build_agent_script(
    project_dir: Path,
    python_path: str,
    ollama_base_url: str,
    ollama_auth_token: str = "",
    env_file: Path | None = None,
) -> str:
    main_path = project_dir / "main.py"
    env_argument = f" --env-file {shell_quote(str(env_file))}" if env_file else ""
    healthcheck_command = _healthcheck_curl_command(ollama_base_url, ollama_auth_token)
    return f"""#!/bin/zsh
set -euo pipefail
cd {shell_quote(str(project_dir))}
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

until {healthcheck_command} >/dev/null 2>&1; do
  sleep 2
done

exec {shell_quote(python_path)} {shell_quote(str(main_path))}{env_argument} poll
"""


def bool_env(value: bool) -> str:
    return "true" if value else "false"


def build_sync_worker_script(
    project_dir: Path,
    python_path: str,
    *,
    sync_worker_seconds: int,
    sync_waiting_enabled: bool,
    sync_trello_done_enabled: bool,
    sync_telegram_poll_enabled: bool,
    final_reply_mode: str,
    env_file: Path | None = None,
) -> str:
    main_path = project_dir / "main.py"
    env_argument = f" --env-file {shell_quote(str(env_file))}" if env_file else ""
    return f"""#!/bin/zsh
set -euo pipefail
cd {shell_quote(str(project_dir))}
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export SYNC_WORKER_SECONDS={shell_quote(str(sync_worker_seconds))}
export SYNC_WAITING_ENABLED={shell_quote(bool_env(sync_waiting_enabled))}
export SYNC_TRELLO_DONE_ENABLED={shell_quote(bool_env(sync_trello_done_enabled))}
export SYNC_TELEGRAM_POLL_ENABLED={shell_quote(bool_env(sync_telegram_poll_enabled))}
export FINAL_REPLY_MODE={shell_quote(final_reply_mode)}

while true; do
  {shell_quote(python_path)} {shell_quote(str(main_path))}{env_argument} automation-outbox-sync --limit 20 || echo "automation-outbox-sync failed with $?"
  {shell_quote(python_path)} {shell_quote(str(main_path))}{env_argument} trello-reply-sync --limit 50 || echo "trello-reply-sync failed with $?"

  if [[ "$SYNC_WAITING_ENABLED" == "true" ]]; then
    {shell_quote(python_path)} {shell_quote(str(main_path))}{env_argument} trello-waiting-sync --limit 50 || echo "trello-waiting-sync failed with $?"
  fi

  if [[ "$SYNC_TRELLO_DONE_ENABLED" == "true" ]]; then
    {shell_quote(python_path)} {shell_quote(str(main_path))}{env_argument} trello-done-sync --limit 50 || echo "trello-done-sync failed with $?"
  fi

  if [[ "$SYNC_TELEGRAM_POLL_ENABLED" == "true" && "$FINAL_REPLY_MODE" == "telegram_approval" ]]; then
    {shell_quote(python_path)} {shell_quote(str(main_path))}{env_argument} telegram-poll --limit 20 || echo "telegram-poll failed with $?"
  fi

  sleep "$SYNC_WORKER_SECONDS"
done
"""


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_launchd_plist(
    *,
    label: str,
    program_arguments: list[str],
    stdout_path: Path,
    stderr_path: Path,
    working_directory: Path,
    keep_alive: bool,
) -> dict:
    return {
        "Label": label,
        "ProgramArguments": program_arguments,
        "RunAtLoad": True,
        "KeepAlive": keep_alive,
        "WorkingDirectory": str(working_directory),
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "EnvironmentVariables": {
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        },
    }


def install_launch_agents(
    project_dir: Path,
    ollama_base_url: str,
    *,
    ollama_auth_token: str = "",
    manage_ollama: bool = False,
    env_file: Path | None = None,
    sync_worker_seconds: int = 60,
    sync_waiting_enabled: bool = True,
    sync_trello_done_enabled: bool = True,
    sync_telegram_poll_enabled: bool = True,
    final_reply_mode: str = "telegram_approval",
) -> LaunchArtifacts:
    python_path = str(project_dir / ".venv" / "bin" / "python")
    if not Path(python_path).exists():
        raise RuntimeError("No encontré la Python de la .venv del proyecto.")

    artifacts = build_launch_artifacts(project_dir)
    launch_working_directory = Path.home()

    if manage_ollama:
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            raise RuntimeError("No encontré `ollama` en PATH.")
        artifacts.ollama_script.write_text(
            build_ollama_script(project_dir, ollama_path), encoding="utf-8"
        )
        ensure_executable(artifacts.ollama_script)
    else:
        artifacts.ollama_script.unlink(missing_ok=True)

    artifacts.agent_script.write_text(
        build_agent_script(
            project_dir,
            python_path,
            ollama_base_url,
            ollama_auth_token,
            env_file,
        ),
        encoding="utf-8",
    )
    artifacts.sync_script.write_text(
        build_sync_worker_script(
            project_dir,
            python_path,
            sync_worker_seconds=sync_worker_seconds,
            sync_waiting_enabled=sync_waiting_enabled,
            sync_trello_done_enabled=sync_trello_done_enabled,
            sync_telegram_poll_enabled=sync_telegram_poll_enabled,
            final_reply_mode=final_reply_mode,
            env_file=env_file,
        ),
        encoding="utf-8",
    )
    ensure_executable(artifacts.agent_script)
    ensure_executable(artifacts.sync_script)

    ollama_plist = None
    if manage_ollama:
        ollama_plist = write_launchd_plist(
            label=OLLAMA_LABEL,
            program_arguments=["/bin/zsh", str(artifacts.ollama_script)],
            stdout_path=artifacts.log_dir / "ollama.stdout.log",
            stderr_path=artifacts.log_dir / "ollama.stderr.log",
            working_directory=launch_working_directory,
            keep_alive=True,
        )
    agent_plist = write_launchd_plist(
        label=AGENT_LABEL,
        program_arguments=["/bin/zsh", str(artifacts.agent_script)],
        stdout_path=artifacts.log_dir / "agent.stdout.log",
        stderr_path=artifacts.log_dir / "agent.stderr.log",
        working_directory=launch_working_directory,
        keep_alive=True,
    )
    sync_plist = write_launchd_plist(
        label=SYNC_LABEL,
        program_arguments=["/bin/zsh", str(artifacts.sync_script)],
        stdout_path=artifacts.log_dir / "sync.stdout.log",
        stderr_path=artifacts.log_dir / "sync.stderr.log",
        working_directory=launch_working_directory,
        keep_alive=True,
    )

    if ollama_plist is not None:
        artifacts.ollama_plist.write_bytes(plistlib.dumps(ollama_plist))
    else:
        artifacts.ollama_plist.unlink(missing_ok=True)
    artifacts.agent_plist.write_bytes(plistlib.dumps(agent_plist))
    artifacts.sync_plist.write_bytes(plistlib.dumps(sync_plist))

    uid = os.getuid()
    for plist in (artifacts.sync_plist, artifacts.agent_plist, artifacts.ollama_plist):
        subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(plist)], check=False)
    plists_to_bootstrap = [artifacts.agent_plist, artifacts.sync_plist]
    if manage_ollama:
        plists_to_bootstrap.insert(0, artifacts.ollama_plist)
    for plist in plists_to_bootstrap:
        subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist)], check=True)

    return artifacts


def uninstall_launch_agents(project_dir: Path) -> LaunchArtifacts:
    artifacts = build_launch_artifacts(project_dir)
    uid = os.getuid()
    for plist in (artifacts.sync_plist, artifacts.agent_plist, artifacts.ollama_plist):
        if plist.exists():
            subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(plist)], check=False)
            plist.unlink(missing_ok=True)
    return artifacts
