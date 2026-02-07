#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_BIN="${RUNTIME_BIN:-container}"
IMAGE_REPO="${IMAGE_REPO:-metaclaw/obsidian-terminal-bot}"
IMAGE_TAG="${IMAGE_TAG:-local}"
WRITE_CLAW="${WRITE_CLAW:-1}"

if ! command -v "$RUNTIME_BIN" >/dev/null 2>&1; then
  echo "runtime binary not found: $RUNTIME_BIN" >&2
  exit 1
fi

TAGGED_IMAGE="$IMAGE_REPO:$IMAGE_TAG"

"$RUNTIME_BIN" build \
  --file "$PROJECT_DIR/image/Dockerfile" \
  --tag "$TAGGED_IMAGE" \
  "$PROJECT_DIR"

LIST_JSON="$($RUNTIME_BIN image list --format json)"
REF="$(printf '%s' "$LIST_JSON" | jq -r --arg tag "$TAGGED_IMAGE" '.[] | select(.reference == $tag or (.reference | endswith("/" + $tag)) or (.reference | endswith($tag))) | .reference' | head -n 1)"
DIGEST="$(printf '%s' "$LIST_JSON" | jq -r --arg tag "$TAGGED_IMAGE" '.[] | select(.reference == $tag or (.reference | endswith("/" + $tag)) or (.reference | endswith($tag))) | .descriptor.digest' | head -n 1)"

if [ -z "$REF" ] || [ -z "$DIGEST" ] || [ "$DIGEST" = "null" ]; then
  echo "failed to resolve built image reference/digest for $TAGGED_IMAGE" >&2
  exit 1
fi

PINNED_REF="${REF%%@*}@$DIGEST"

# Ensure the local store has an explicit digest reference to avoid remote pull attempts.
"$RUNTIME_BIN" image tag "$TAGGED_IMAGE" "$PINNED_REF" >/dev/null

echo "built_image_tag: $TAGGED_IMAGE"
echo "resolved_reference: $REF"
echo "resolved_digest: $DIGEST"
echo "pinned_runtime_image: $PINNED_REF"

if [ "$WRITE_CLAW" = "1" ]; then
  CLAW_PATH="$PROJECT_DIR/agent.claw"
  if [ ! -f "$CLAW_PATH" ]; then
    echo "agent.claw not found: $CLAW_PATH" >&2
    exit 1
  fi
  TMP_FILE="$(mktemp)"
  awk -v img="$PINNED_REF" '
    BEGIN {replaced=0}
    {
      if ($0 ~ /^[[:space:]]*image:[[:space:]]*/) {
        sub(/image:[[:space:]]*.*/, "image: " img)
        replaced=1
      }
      print
    }
    END {
      if (replaced == 0) {
        exit 7
      }
    }
  ' "$CLAW_PATH" > "$TMP_FILE" || {
    rm -f "$TMP_FILE"
    echo "failed to patch runtime.image in agent.claw" >&2
    exit 1
  }
  mv "$TMP_FILE" "$CLAW_PATH"
  echo "updated agent.claw runtime.image"
fi
