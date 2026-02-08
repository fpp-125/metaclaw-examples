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

resolve_pinned_image() {
  local ref=""
  local digest=""
  local fallback_ref="${TAGGED_IMAGE%%@*}"

  case "$RUNTIME_BIN" in
    podman|docker)
      ref="$("$RUNTIME_BIN" image inspect "$TAGGED_IMAGE" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
      ref="$(printf '%s' "$ref" | head -n 1 | tr -d '\r')"
      if [ -n "$ref" ] && [ "$ref" != "<no value>" ] && [ "$ref" != "null" ]; then
        digest="${ref##*@}"
      else
        digest="$("$RUNTIME_BIN" image inspect "$TAGGED_IMAGE" --format '{{.Digest}}' 2>/dev/null || true)"
        digest="$(printf '%s' "$digest" | head -n 1 | tr -d '\r')"
        ref=""
      fi
      ;;
    container)
      if ! command -v jq >/dev/null 2>&1; then
        echo "jq not found: required to parse container image inspect output" >&2
        return 1
      fi
      local inspect_json
      inspect_json="$("$RUNTIME_BIN" image inspect "$TAGGED_IMAGE")"
      digest="$(printf '%s' "$inspect_json" | jq -r '.[0].index.digest // empty' | head -n 1)"
      ref="$(printf '%s' "$inspect_json" | jq -r '.[0].name // empty' | head -n 1)"
      ;;
    *)
      if ! command -v jq >/dev/null 2>&1; then
        echo "jq not found: required to parse image list output for runtime $RUNTIME_BIN" >&2
        return 1
      fi
      local list_json
      list_json="$("$RUNTIME_BIN" image list --format json)"
      ref="$(printf '%s' "$list_json" | jq -r --arg tag "$TAGGED_IMAGE" '
        .[]?
        | . as $img
        | (if (($img.reference // null) | type) == "string" then $img.reference else "" end) as $r
        | select($r != "" and ($r == $tag or ($r | endswith("/" + $tag)) or ($r | endswith($tag))))
        | $r
      ' | head -n 1)"
      digest="$(printf '%s' "$list_json" | jq -r --arg ref "$ref" '
        .[]?
        | select((.reference // "") == $ref)
        | (.descriptor.digest // .Digest // empty)
      ' | head -n 1)"
      ;;
  esac

  if [ -z "$digest" ] || [ "$digest" = "null" ] || [ "$digest" = "<no value>" ]; then
    echo "failed to resolve digest for built image $TAGGED_IMAGE using runtime $RUNTIME_BIN" >&2
    return 1
  fi

  if [ -z "$ref" ] || [ "$ref" = "null" ] || [ "$ref" = "<no value>" ]; then
    ref="$fallback_ref"
  fi
  ref="${ref%%@*}"

  RESOLVED_REFERENCE="$ref"
  RESOLVED_DIGEST="$digest"
  PINNED_REF="${RESOLVED_REFERENCE}@${RESOLVED_DIGEST}"
}

RESOLVED_REFERENCE=""
RESOLVED_DIGEST=""
PINNED_REF=""
resolve_pinned_image

# Ensure the local store has an explicit digest reference to avoid remote pull attempts.
if [ "$RUNTIME_BIN" != "docker" ]; then
  if ! "$RUNTIME_BIN" image tag "$TAGGED_IMAGE" "$PINNED_REF" >/dev/null 2>&1; then
    echo "warning: unable to tag digest reference in local runtime store: $PINNED_REF" >&2
  fi
fi

echo "built_image_tag: $TAGGED_IMAGE"
echo "resolved_reference: $RESOLVED_REFERENCE"
echo "resolved_digest: $RESOLVED_DIGEST"
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
