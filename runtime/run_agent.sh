#!/bin/zsh
set -euo pipefail
cd '/Users/ivanrodriguez/Documents/Automatizaciones/slack-personal-agent'
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

until curl -sf 'http://127.0.0.1:11434'/api/tags >/dev/null 2>&1; do
  sleep 2
done

exec '/Users/ivanrodriguez/Documents/Automatizaciones/slack-personal-agent/.venv/bin/python' '/Users/ivanrodriguez/Documents/Automatizaciones/slack-personal-agent/main.py' poll
