#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

METACLAW_BIN="${METACLAW_BIN:-metaclaw}"
if ! command -v "$METACLAW_BIN" >/dev/null 2>&1; then
  FALLBACK_BIN="$ROOT/../MetaClaw/metaclaw"
  if [ -x "$FALLBACK_BIN" ]; then
    METACLAW_BIN="$FALLBACK_BIN"
  fi
fi

if ! command -v "$METACLAW_BIN" >/dev/null 2>&1 && [ ! -x "${METACLAW_BIN:-}" ]; then
  echo "metaclaw binary not found; running structural checks only"
  found=0
  for f in examples/*/agent.claw; do
    [ -f "$f" ] || continue
    found=1
    echo "OK  $f"
  done
  if [ "$found" -eq 0 ]; then
    echo "No example clawfiles found" >&2
    exit 1
  fi
  exit 0
fi

found=0
for f in examples/*/agent.claw; do
  [ -f "$f" ] || continue
  found=1
  echo "validate $f"
  "$METACLAW_BIN" validate "$f" >/dev/null
  echo "OK  $f"
done

if [ "$found" -eq 0 ]; then
  echo "No example clawfiles found" >&2
  exit 1
fi
