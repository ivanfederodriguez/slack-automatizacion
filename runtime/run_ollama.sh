#!/bin/zsh
set -euo pipefail
cd '/Users/ivanrodriguez/Documents/Automatizaciones/slack-personal-agent'
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
exec '/usr/local/bin/ollama' serve
