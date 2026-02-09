#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load project-local secrets (never commit .env)
if [ -f "$PROJECT_DIR/.env" ]; then
  set -a
  . "$PROJECT_DIR/.env"
  set +a
fi

export BOT_RENDER_MODE="${BOT_RENDER_MODE:-glow}"
export BOT_NETWORK_MODE="${BOT_NETWORK_MODE:-none}"
exec python3 "$PROJECT_DIR/chat_tui.py" "$@"
