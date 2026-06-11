from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


OLLAMA_LABEL = "com.ivanrodriguez.slack-agent.ollama"
AGENT_LABEL = "com.ivanrodriguez.slack-agent.agent"


@dataclass(frozen=True)
class LaunchArtifacts:
    ollama_plist: Path
    agent_plist: Path
    ollama_script: Path
    agent_script: Path
    log_dir: Path


def ensure_executable(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(current_mode | 0o111)


def build_launch_artifacts(project_dir: Path) -> LaunchArtifacts:
    support_dir = project_dir / "runtime"
    launch_dir = Path.home() / "Library" / "LaunchAgents"
    log_dir = support_dir / "logs"
    support_dir.mkdir(parents=True, exist_ok=True)
    launch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return LaunchArtifacts(
        ollama_plist=launch_dir / f"{OLLAMA_LABEL}.plist",
        agent_plist=launch_dir / f"{AGENT_LABEL}.plist",
        ollama_script=support_dir / "run_ollama.sh",
        agent_script=support_dir / "run_agent.sh",
        log_dir=log_dir,
    )


def build_ollama_script(project_dir: Path, ollama_path: str) -> str:
    return f"""#!/bin/zsh
set -euo pipefail
cd {shell_quote(str(project_dir))}
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
exec {shell_quote(ollama_path)} serve
"""


def build_agent_script(project_dir: Path, python_path: str, ollama_base_url: str) -> str:
    main_path = project_dir / "main.py"
    return f"""#!/bin/zsh
set -euo pipefail
cd {shell_quote(str(project_dir))}
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

for attempt in {{1..60}}; do
  if curl -sf {shell_quote(ollama_base_url.rstrip('/'))}/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

exec {shell_quote(python_path)} {shell_quote(str(main_path))} poll
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


def install_launch_agents(project_dir: Path, ollama_base_url: str) -> LaunchArtifacts:
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        raise RuntimeError("No encontré `ollama` en PATH.")

    python_path = str(project_dir / ".venv" / "bin" / "python")
    if not Path(python_path).exists():
        raise RuntimeError("No encontré la Python de la .venv del proyecto.")

    artifacts = build_launch_artifacts(project_dir)

    artifacts.ollama_script.write_text(build_ollama_script(project_dir, ollama_path), encoding="utf-8")
    artifacts.agent_script.write_text(build_agent_script(project_dir, python_path, ollama_base_url), encoding="utf-8")
    ensure_executable(artifacts.ollama_script)
    ensure_executable(artifacts.agent_script)

    ollama_plist = write_launchd_plist(
        label=OLLAMA_LABEL,
        program_arguments=[
            "/bin/zsh",
            "-lc",
            f"curl -sf {ollama_base_url.rstrip('/')}/api/tags >/dev/null 2>&1 || exec {shell_quote(ollama_path)} serve",
        ],
        stdout_path=artifacts.log_dir / "ollama.stdout.log",
        stderr_path=artifacts.log_dir / "ollama.stderr.log",
        working_directory=project_dir,
        keep_alive=False,
    )
    agent_plist = write_launchd_plist(
        label=AGENT_LABEL,
        program_arguments=[python_path, str(project_dir / "main.py"), "poll"],
        stdout_path=artifacts.log_dir / "agent.stdout.log",
        stderr_path=artifacts.log_dir / "agent.stderr.log",
        working_directory=project_dir,
        keep_alive=True,
    )

    artifacts.ollama_plist.write_bytes(plistlib.dumps(ollama_plist))
    artifacts.agent_plist.write_bytes(plistlib.dumps(agent_plist))

    uid = os.getuid()
    for plist in (artifacts.ollama_plist, artifacts.agent_plist):
        subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(plist)], check=False)
        subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist)], check=True)

    return artifacts


def uninstall_launch_agents(project_dir: Path) -> LaunchArtifacts:
    artifacts = build_launch_artifacts(project_dir)
    uid = os.getuid()
    for plist in (artifacts.agent_plist, artifacts.ollama_plist):
        if plist.exists():
            subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(plist)], check=False)
            plist.unlink(missing_ok=True)
    return artifacts
