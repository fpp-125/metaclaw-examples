#!/usr/bin/env python3
"""Improved terminal UI for MetaClaw Obsidian bot."""

from __future__ import annotations

import datetime as dt
import difflib
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import termios
import textwrap
import time
import tty
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    import readline  # noqa: F401
except Exception:  # pragma: no cover
    readline = None


PROJECT_DIR = Path(__file__).resolve().parent
AGENT_FILE = PROJECT_DIR / "agent.claw"
HOST_DATA_DIR = Path(os.getenv("BOT_HOST_DATA_DIR", str(Path.home() / ".metaclaw" / "obsidian-terminal-bot")))
HOST_CONFIG_DIR = Path(os.getenv("BOT_HOST_CONFIG_DIR", str(HOST_DATA_DIR / "config")))
HOST_LOG_DIR = Path(os.getenv("BOT_HOST_LOG_DIR", str(HOST_DATA_DIR / "logs")))
HOST_RUNTIME_DIR = Path(os.getenv("BOT_HOST_RUNTIME_DIR", str(HOST_DATA_DIR / "runtime")))
HOST_WORKSPACE_DIR = Path(os.getenv("BOT_HOST_WORKSPACE_DIR", str(HOST_DATA_DIR / "workspace")))
STATE_DIR = Path(os.getenv("BOT_STATE_DIR", str(HOST_DATA_DIR / "state")))
RUNTIME_DIR = HOST_RUNTIME_DIR
REQUEST_FILE = RUNTIME_DIR / "request.txt"
RESPONSE_FILE = RUNTIME_DIR / "response.txt"
HISTORY_FILE = RUNTIME_DIR / "history.json"
SESSION_FILE = RUNTIME_DIR / "session.json"
INPUT_HISTORY_FILE = RUNTIME_DIR / ".cli_input_history"
RUN_LOG_FILE = Path("/tmp/metaclaw_chat_run.log")
EFFECTIVE_AGENT_FILE = RUNTIME_DIR / "agent.effective.claw"
DEFAULTS_FILE = HOST_CONFIG_DIR / "ui.defaults.json"
LLM_CONFIG_FILE = HOST_CONFIG_DIR / "llm.config.json"
WRITE_AUDIT_FILE = HOST_LOG_DIR / "write_audit.jsonl"
LAST_RESPONSE_MD_FILE = RUNTIME_DIR / "last_response.md"
AGENTS_CONTEXT_FILE = RUNTIME_DIR / "agents_context.md"
DOTENV_FILE = PROJECT_DIR / ".env"
AGENTS_DIR = PROJECT_DIR / "agents"
AGENTS_FILE = AGENTS_DIR / "AGENTS.md"
SOUL_FILE = AGENTS_DIR / "soul.md"

SESSION_ROOT_DIR = HOST_RUNTIME_DIR / "sessions"
CURRENT_SESSION_FILE = HOST_CONFIG_DIR / "current_session.json"
SESSION_MANAGED_FILES = (
    "request.txt",
    "response.txt",
    "history.json",
    "session.json",
    "agents_context.md",
    "last_response.md",
)

DEFAULT_METACLAW_BIN = "metaclaw"
FALLBACK_METACLAW_BINS = (
    PROJECT_DIR / "metaclaw",
    PROJECT_DIR.parent / "metaclaw",
    PROJECT_DIR.parents[2] / "MetaClaw" / "metaclaw",
    PROJECT_DIR.parents[2] / "metaclaw" / "metaclaw",
)

ALLOWED_SAVE_ROOTS = ("Research", "Learning")
DEFAULT_VALUES = {
    "network_mode": "none",
    "render_mode": "glow",
    "vault_access": "ro",
    "retrieval_scope": "limited",
    "write_confirm_mode": "enter_once",
    "save_default_dir": "Research/Market-Reports",
    # Where you "land" after a response:
    # - stay: print inline (normal scrollback)
    # - start: open a pager at the top
    # - end: open a pager at the bottom
    # - input: keep you at the prompt; store response for `/show`
    "output_focus": "start",
    # Tool-loop budgets (container-side env). Larger reorganizations need more steps.
    "tool_max_steps": 16,
    "tool_max_calls_per_step": 12,
}

NETWORK_CHOICES = [
    ("none", "none"),
    ("outbound", "out"),
]
VAULT_CHOICES = [
    ("ro", "vault ro (recommended)"),
    ("rw", "vault rw (less safe)"),
]
RENDER_CHOICES = [
    ("plain", "plain"),
    ("glow", "glow"),
]
SCOPE_CHOICES = [
    ("limited", "limited"),
    ("all", "all"),
]
CONFIRM_CHOICES = [
    ("enter_once", "once"),
    ("diff_yes", "diff"),
    ("allowlist_auto", "auto"),
]
FOCUS_CHOICES = [
    ("start", "start"),
    ("end", "end"),
    ("stay", "stay"),
    ("input", "input"),
]

LLM_PROVIDER_CATALOG: dict[str, dict] = {
    "gemini_openai": {
        "id": "gemini_openai",
        "label": "Gemini",
        "engineProvider": "gemini_openai",
        "baseURL": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "keyEnv": "GEMINI_API_KEY",
        "models": [
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.0-flash",
            "gemini-2.0-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    },
    "openai": {
        "id": "openai",
        "label": "OpenAI",
        "engineProvider": "openai_compatible",
        "baseURL": "https://api.openai.com/v1",
        "keyEnv": "OPENAI_API_KEY",
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
            "o1-mini",
            "o1",
        ],
    },
    "anthropic": {
        "id": "anthropic",
        "label": "Anthropic",
        "engineProvider": "anthropic",
        "baseURL": "https://api.anthropic.com/v1",
        "keyEnv": "ANTHROPIC_API_KEY",
        "models": [
            "claude-3-5-haiku-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
    },
}


@dataclass
class SessionState:
    network_mode: str
    render_mode: str
    vault_access: str
    retrieval_scope: str
    write_confirm_mode: str
    save_default_dir: str
    output_focus: str
    tool_max_steps: int
    tool_max_calls_per_step: int


def supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


USE_COLOR = supports_color()


def c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def width() -> int:
    return max(72, shutil.get_terminal_size(fallback=(100, 30)).columns)


def boxed(title: str, rows: list[str]) -> None:
    w = width()
    top = c("=" * w, "2")
    print(top)
    print(c(f" {title}", "1;36"))
    print(c("-" * w, "2"))
    for row in rows:
        print(row)
    print(c("=" * w, "2"))


def setup_readline() -> None:
    if "readline" not in globals() or readline is None:
        return
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if INPUT_HISTORY_FILE.exists():
            readline.read_history_file(str(INPUT_HISTORY_FILE))
        readline.set_history_length(800)
    except Exception:
        pass


def save_readline_history() -> None:
    if "readline" not in globals() or readline is None:
        return
    try:
        readline.write_history_file(str(INPUT_HISTORY_FILE))
    except Exception:
        pass


def resolve_metaclaw_bin() -> str:
    def is_executable_file(path: Path) -> bool:
        return path.is_file() and os.access(str(path), os.X_OK)

    def find_executable(path: Path) -> Path | None:
        # Accept either a direct binary path or a repo directory containing ./bin/metaclaw.
        if is_executable_file(path):
            return path
        if path.is_dir():
            for sub in (
                path / "bin" / "metaclaw",
                path / "metaclaw",
            ):
                if is_executable_file(sub):
                    return sub
        return None

    metaclaw_bin = os.getenv("METACLAW_BIN", DEFAULT_METACLAW_BIN).strip().strip("'\"")
    resolved = shutil.which(metaclaw_bin) if metaclaw_bin else None
    if resolved:
        return resolved
    if metaclaw_bin:
        found = find_executable(Path(metaclaw_bin))
        if found:
            return str(found)
    for candidate in FALLBACK_METACLAW_BINS:
        found = find_executable(candidate)
        if found:
            return str(found)
    print("metaclaw binary not found.", file=sys.stderr)
    print("Build it (from the metaclaw repo): go build -o ./bin/metaclaw ./cmd/metaclaw", file=sys.stderr)
    print("Then set METACLAW_BIN=/absolute/path/to/bin/metaclaw", file=sys.stderr)
    sys.exit(1)


def ensure_paths() -> None:
    HOST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    HOST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    HOST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    HOST_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure we have an active runtime session with stable symlinks:
    # /runtime/request.txt -> /runtime/sessions/<id>/request.txt etc.
    ensure_runtime_session()

    local_config_dir = PROJECT_DIR / "config"
    if local_config_dir.exists() and not any(HOST_CONFIG_DIR.iterdir()):
        for child in local_config_dir.iterdir():
            if child.is_file():
                shutil.copy2(child, HOST_CONFIG_DIR / child.name)


def normalize_network_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode == "out":
        return "outbound"
    if mode not in {"none", "outbound"}:
        return DEFAULT_VALUES["network_mode"]
    return mode


def normalize_render_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in {"plain", "glow"}:
        mode = DEFAULT_VALUES["render_mode"]
    if mode == "glow" and shutil.which("glow") is None:
        return "plain"
    return mode


def normalize_vault_access(value: str) -> str:
    value = value.strip().lower()
    aliases = {
        "read-only": "ro",
        "readonly": "ro",
        "read_only": "ro",
        "read-write": "rw",
        "readwrite": "rw",
        "read_write": "rw",
    }
    value = aliases.get(value, value)
    if value not in {"ro", "rw"}:
        return DEFAULT_VALUES["vault_access"]
    return value


def normalize_scope(scope: str) -> str:
    scope = scope.strip().lower()
    if scope not in {"limited", "all"}:
        return DEFAULT_VALUES["retrieval_scope"]
    return scope


def normalize_confirm_mode(mode: str) -> str:
    mode = mode.strip().lower()
    aliases = {
        "once": "enter_once",
        "diff": "diff_yes",
        "auto": "allowlist_auto",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"enter_once", "diff_yes", "allowlist_auto"}:
        return DEFAULT_VALUES["write_confirm_mode"]
    return mode


def normalize_focus(value: str) -> str:
    v = (value or "").strip().lower()
    if v not in {"start", "end", "stay", "input"}:
        return DEFAULT_VALUES["output_focus"]
    return v


def normalize_tool_max_steps(value: int | str | None) -> int:
    try:
        n = int(str(value).strip())
    except Exception:
        n = int(DEFAULT_VALUES["tool_max_steps"])
    if n < 1:
        n = 1
    if n > 128:
        n = 128
    return n


def normalize_tool_max_calls_per_step(value: int | str | None) -> int:
    try:
        n = int(str(value).strip())
    except Exception:
        n = int(DEFAULT_VALUES["tool_max_calls_per_step"])
    if n < 1:
        n = 1
    if n > 24:
        n = 24
    return n


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,64}$")


def sanitize_session_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    # Keep it filesystem-friendly and readable.
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        return ""
    if len(s) > 64:
        s = s[:64]
    if not _SESSION_ID_RE.match(s):
        return ""
    return s


def get_current_session_id() -> str:
    if not CURRENT_SESSION_FILE.exists():
        return "default"
    try:
        payload = json.loads(CURRENT_SESSION_FILE.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return "default"
    if not isinstance(payload, dict):
        return "default"
    sid = sanitize_session_id(str(payload.get("id", "")).strip())
    return sid or "default"


def set_current_session_id(session_id: str) -> None:
    session_id = sanitize_session_id(session_id) or "default"
    CURRENT_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_SESSION_FILE.write_text(json.dumps({"id": session_id}, indent=2) + "\n", encoding="utf-8")


def is_symlink(p: Path) -> bool:
    try:
        return p.is_symlink()
    except Exception:
        return False


def ensure_runtime_session() -> str:
    """Ensure runtime session directories and stable symlinks exist.

    The container always reads /runtime/{request,response,history,...}. We keep those as symlinks
    into /runtime/sessions/<id>/... so users can start a fresh session without copying projects.
    """
    SESSION_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    sid = get_current_session_id()
    activate_runtime_session(sid, migrate_existing=True)
    return sid


def activate_runtime_session(session_id: str, *, migrate_existing: bool) -> str:
    sid = sanitize_session_id(session_id)
    if not sid:
        # Timestamp-based fallback.
        sid = dt.datetime.now(dt.timezone.utc).strftime("s-%Y%m%dT%H%M%SZ")
    session_dir = SESSION_ROOT_DIR / sid
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create target files first.
    for name in SESSION_MANAGED_FILES:
        p = session_dir / name
        if p.exists():
            continue
        if name.endswith(".json"):
            p.write_text("[]\n" if name == "history.json" else "{}\n", encoding="utf-8")
        else:
            p.write_text("", encoding="utf-8")

    # For older projects, migrate existing non-symlink runtime files into a legacy session
    # to avoid data loss.
    legacy_dir: Path | None = None
    if migrate_existing:
        for name in SESSION_MANAGED_FILES:
            link_path = RUNTIME_DIR / name
            if not link_path.exists() or is_symlink(link_path):
                continue
            if legacy_dir is None:
                legacy_dir = SESSION_ROOT_DIR / ("legacy-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
                legacy_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(link_path), str(legacy_dir / name))
            except Exception:
                # Best-effort; if move fails, leave it and proceed.
                pass

    # Point stable paths to the current session.
    for name in SESSION_MANAGED_FILES:
        link_path = RUNTIME_DIR / name
        target_rel = Path("sessions") / sid / name
        try:
            if link_path.exists() or is_symlink(link_path):
                link_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            link_path.symlink_to(target_rel)
        except Exception:
            # Symlinks may be unavailable on some systems; fall back to copying.
            try:
                shutil.copy2(str(session_dir / name), str(link_path))
            except Exception:
                pass

    set_current_session_id(sid)
    return sid


def sanitize_default_dir(value: str) -> str:
    cleaned = sanitize_vault_relative_path(value, require_md=False)
    first = cleaned.parts[0] if cleaned.parts else ""
    if first not in ALLOWED_SAVE_ROOTS:
        raise ValueError(f"default dir must start with one of {', '.join(ALLOWED_SAVE_ROOTS)}")
    return cleaned.as_posix()


def load_defaults() -> dict[str, str]:
    out = dict(DEFAULT_VALUES)

    # Align with the current agent.claw unless user overrides in ui.defaults.json.
    ro = read_vault_readonly_from_agent()
    if ro is False:
        out["vault_access"] = "rw"
    elif ro is True:
        out["vault_access"] = "ro"

    if not DEFAULTS_FILE.exists():
        return out
    try:
        payload = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out

    try:
        if isinstance(payload.get("network_mode"), str):
            out["network_mode"] = normalize_network_mode(payload["network_mode"])
        if isinstance(payload.get("render_mode"), str):
            out["render_mode"] = normalize_render_mode(payload["render_mode"])
        if isinstance(payload.get("vault_access"), str):
            out["vault_access"] = normalize_vault_access(payload["vault_access"])
        if isinstance(payload.get("retrieval_scope"), str):
            out["retrieval_scope"] = normalize_scope(payload["retrieval_scope"])
        if isinstance(payload.get("write_confirm_mode"), str):
            out["write_confirm_mode"] = normalize_confirm_mode(payload["write_confirm_mode"])
        if isinstance(payload.get("save_default_dir"), str):
            out["save_default_dir"] = sanitize_default_dir(payload["save_default_dir"])
        if isinstance(payload.get("output_focus"), str):
            out["output_focus"] = normalize_focus(payload["output_focus"])
        if payload.get("tool_max_steps") is not None:
            out["tool_max_steps"] = normalize_tool_max_steps(payload.get("tool_max_steps"))
        if payload.get("tool_max_calls_per_step") is not None:
            out["tool_max_calls_per_step"] = normalize_tool_max_calls_per_step(payload.get("tool_max_calls_per_step"))
    except Exception:
        return dict(DEFAULT_VALUES)
    return out


def save_defaults(values: dict[str, str]) -> None:
    payload = {
        "network_mode": normalize_network_mode(values.get("network_mode", DEFAULT_VALUES["network_mode"])),
        "render_mode": normalize_render_mode(values.get("render_mode", DEFAULT_VALUES["render_mode"])),
        "vault_access": normalize_vault_access(values.get("vault_access", DEFAULT_VALUES["vault_access"])),
        "retrieval_scope": normalize_scope(values.get("retrieval_scope", DEFAULT_VALUES["retrieval_scope"])),
        "write_confirm_mode": normalize_confirm_mode(values.get("write_confirm_mode", DEFAULT_VALUES["write_confirm_mode"])),
        "save_default_dir": sanitize_default_dir(values.get("save_default_dir", DEFAULT_VALUES["save_default_dir"])),
        "output_focus": normalize_focus(values.get("output_focus", DEFAULT_VALUES["output_focus"])),
        "tool_max_steps": normalize_tool_max_steps(values.get("tool_max_steps", DEFAULT_VALUES["tool_max_steps"])),
        "tool_max_calls_per_step": normalize_tool_max_calls_per_step(values.get("tool_max_calls_per_step", DEFAULT_VALUES["tool_max_calls_per_step"])),
    }
    DEFAULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULTS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_llm_config() -> dict:
    gem = LLM_PROVIDER_CATALOG["gemini_openai"]
    model = gem["models"][0]
    return {
        "schemaVersion": 1,
        "providers": [
            {
                "id": gem["id"],
                "label": gem["label"],
                "engineProvider": gem["engineProvider"],
                "baseURL": gem["baseURL"],
                "keyEnv": gem["keyEnv"],
                "models": list(gem["models"]),
                "selectedModels": [model],
            }
        ],
        "default": {"providerId": gem["id"], "model": model},
    }


def load_llm_config() -> dict:
    cfg = default_llm_config()
    if not LLM_CONFIG_FILE.exists():
        return cfg
    try:
        payload = json.loads(LLM_CONFIG_FILE.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return cfg
    if not isinstance(payload, dict):
        return cfg

    # Shallow-merge known fields.
    try:
        if isinstance(payload.get("schemaVersion"), int):
            cfg["schemaVersion"] = payload["schemaVersion"]
        if isinstance(payload.get("providers"), list):
            cfg["providers"] = payload["providers"]
        if isinstance(payload.get("default"), dict):
            cfg["default"] = payload["default"]
    except Exception:
        return default_llm_config()

    return normalize_llm_config(cfg)


def save_llm_config(cfg: dict) -> None:
    cfg = normalize_llm_config(cfg)
    LLM_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LLM_CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_llm_config(cfg: dict) -> dict:
    # Ensure a minimal, stable shape.
    out = {"schemaVersion": 1, "providers": [], "default": {"providerId": "", "model": ""}}
    if isinstance(cfg, dict):
        if isinstance(cfg.get("schemaVersion"), int):
            out["schemaVersion"] = int(cfg["schemaVersion"]) or 1
        if isinstance(cfg.get("providers"), list):
            out["providers"] = [p for p in cfg["providers"] if isinstance(p, dict)]
        if isinstance(cfg.get("default"), dict):
            out["default"] = dict(cfg["default"])

    # Fill in from catalog where possible.
    normalized_providers: list[dict] = []
    for p in out["providers"]:
        pid = str(p.get("id", "")).strip()
        if not pid:
            continue
        cat = LLM_PROVIDER_CATALOG.get(pid, {})
        label = str(p.get("label") or cat.get("label") or pid).strip()
        engine_provider = str(p.get("engineProvider") or cat.get("engineProvider") or "").strip()
        base_url = str(p.get("baseURL") or cat.get("baseURL") or "").strip()
        key_env = str(p.get("keyEnv") or cat.get("keyEnv") or "").strip()
        models = p.get("models")
        if not isinstance(models, list) or not models:
            models = list(cat.get("models") or [])
        models = [str(m).strip() for m in models if str(m).strip()]
        # De-dupe while preserving order.
        seen = set()
        models = [m for m in models if not (m in seen or seen.add(m))]

        selected = p.get("selectedModels")
        if not isinstance(selected, list) or not selected:
            selected = [p.get("model")] if isinstance(p.get("model"), str) else []
        selected = [str(m).strip() for m in selected if str(m).strip()]
        selected = [m for m in selected if m in models] or (models[:1] if models else [])

        normalized_providers.append(
            {
                "id": pid,
                "label": label,
                "engineProvider": engine_provider,
                "baseURL": base_url,
                "keyEnv": key_env,
                "models": models,
                "selectedModels": selected,
            }
        )

    if not normalized_providers:
        return default_llm_config()
    out["providers"] = normalized_providers

    # Default selection.
    d = out.get("default") if isinstance(out.get("default"), dict) else {}
    default_pid = str(d.get("providerId", "")).strip()
    default_model = str(d.get("model", "")).strip()
    if default_pid not in {p["id"] for p in normalized_providers}:
        default_pid = normalized_providers[0]["id"]
    provider = next((p for p in normalized_providers if p["id"] == default_pid), normalized_providers[0])
    if default_model not in set(provider.get("selectedModels") or []):
        default_model = (provider.get("selectedModels") or provider.get("models") or [""])[0]
    out["default"] = {"providerId": default_pid, "model": default_model}
    return out


def dotenv_read(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def dotenv_upsert(path: Path, updates: dict[str, str]) -> None:
    updates = {str(k).strip(): str(v).strip() for k, v in updates.items() if str(k).strip()}
    if not updates:
        return
    if any("\n" in v or "\r" in v for v in updates.values()):
        raise ValueError("dotenv values cannot contain newlines")

    lines: list[str] = []
    existing: dict[str, int] = {}
    if path.exists():
        raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, raw in enumerate(raw_lines):
            lines.append(raw)
            s = raw.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k = s.split("=", 1)[0].strip()
            if k and k not in existing:
                existing[k] = i
    else:
        lines = ["# Runtime-only secrets (never commit actual values)"]

    for k, v in updates.items():
        if k in existing:
            lines[existing[k]] = f"{k}={v}"
        else:
            lines.append(f"{k}={v}")

    if lines and lines[-1].strip() != "":
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def write_agents_context() -> None:
    parts: list[str] = []
    try:
        if AGENTS_FILE.exists():
            parts.append("# AGENTS.md\n\n" + AGENTS_FILE.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:
        pass
    try:
        if SOUL_FILE.exists():
            parts.append("# soul.md\n\n" + SOUL_FILE.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:
        pass
    if not parts:
        return
    AGENTS_CONTEXT_FILE.write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")


def get_key_env_name() -> str:
    # Backward compatible behavior:
    # - If no llm.config.json exists yet, fall back to the legacy single-env name (LLM_KEY_ENV).
    # - If llm.config.json exists, prefer the configured default provider key env.
    if not LLM_CONFIG_FILE.exists():
        return os.getenv("LLM_KEY_ENV", "OPENAI_FORMAT_API_KEY")
    cfg = load_llm_config()
    default_pid = str((cfg.get("default") or {}).get("providerId", "")).strip()
    for p in cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []:
        if isinstance(p, dict) and str(p.get("id", "")).strip() == default_pid:
            key_env = str(p.get("keyEnv", "")).strip()
            if key_env and os.getenv(key_env):
                return key_env
    # If the configured env name isn't set, prefer the session's generic key env
    # (set by chat.sh / quickstart) so provider switching "just works".
    fallback = os.getenv("LLM_KEY_ENV", "").strip() or "OPENAI_FORMAT_API_KEY"
    return fallback


def get_web_key_env_name() -> str:
    return os.getenv("TAVILY_KEY_ENV", "TAVILY_API_KEY")


def get_runtime_target() -> str:
    return os.getenv("RUNTIME_TARGET", "apple_container")


def network_mode_label(mode: str) -> str:
    return "out" if mode == "outbound" else "none"


def vault_access_label(mode: str) -> str:
    return "rw" if normalize_vault_access(mode) == "rw" else "ro"


def has_glow() -> bool:
    return shutil.which("glow") is not None


def confirm_mode_label(mode: str) -> str:
    labels = {
        "enter_once": "once",
        "diff_yes": "diff",
        "allowlist_auto": "auto",
    }
    return labels.get(mode, mode)


def focus_label(mode: str) -> str:
    return normalize_focus(mode)


def has_less() -> bool:
    return shutil.which("less") is not None


def key_is_ready() -> bool:
    attempts = llm_attempts_all()
    if not attempts:
        return False
    return pick_first_set_env_key(attempts[0]) is not None


def web_key_is_ready() -> bool:
    return bool(os.getenv(get_web_key_env_name()))


def llm_attempts_all() -> list[dict]:
    """Return ordered LLM (provider, model) attempts for this session.

    The first entry is always the default. Additional entries are fallbacks.
    Each attempt dict contains:
      - providerId, providerLabel
      - engineProvider, baseURL, model
      - keyEnv (host env var name holding the key)
    """
    if not LLM_CONFIG_FILE.exists():
        # Legacy mode: use whatever agent.claw declares, but read the key from a single host env name.
        return [
            {
                "providerId": "legacy",
                "providerLabel": "agent.claw",
                "engineProvider": "",
                "baseURL": "",
                "model": "",
                "keyEnv": get_key_env_name(),
            }
        ]

    cfg = load_llm_config()
    providers = cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []
    default = cfg.get("default") if isinstance(cfg.get("default"), dict) else {}
    default_pid = str(default.get("providerId", "")).strip()
    default_model = str(default.get("model", "")).strip()

    attempts: list[dict] = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", "")).strip()
        label = str(p.get("label", "")).strip() or pid
        engine_provider = str(p.get("engineProvider", "")).strip()
        base_url = str(p.get("baseURL", "")).strip()
        key_env = str(p.get("keyEnv", "")).strip()
        selected = p.get("selectedModels") if isinstance(p.get("selectedModels"), list) else []
        selected_models = [str(m).strip() for m in selected if str(m).strip()]
        if not selected_models:
            continue

        # Default first, then remaining models.
        ordered_models = selected_models[:]
        if pid == default_pid and default_model in ordered_models:
            ordered_models.remove(default_model)
            ordered_models.insert(0, default_model)

        for m in ordered_models:
            attempts.append(
                {
                    "providerId": pid,
                    "providerLabel": label,
                    "engineProvider": engine_provider,
                    "baseURL": base_url,
                    "model": m,
                    "keyEnv": key_env,
                }
            )

    # Ensure default combo is first overall.
    if default_pid and default_model:
        for i, a in enumerate(attempts):
            if a["providerId"] == default_pid and a["model"] == default_model:
                if i != 0:
                    attempts.insert(0, attempts.pop(i))
                break
    return attempts


def llm_attempts_ready() -> list[dict]:
    return [a for a in llm_attempts_all() if pick_first_set_env_key(a) is not None]


def key_env_candidates(preferred: str | None) -> list[str]:
    # Try the provider-specific env name first, but allow quickstarts to use one
    # provider-agnostic env name for all OpenAI-compatible endpoints.
    out: list[str] = []
    preferred = (preferred or "").strip()
    if preferred:
        out.append(preferred)
    generic = os.getenv("LLM_KEY_ENV", "").strip()
    if generic and generic not in out:
        out.append(generic)
    if "OPENAI_FORMAT_API_KEY" not in out:
        out.append("OPENAI_FORMAT_API_KEY")
    return out


def pick_first_set_env_key(attempt: dict) -> str | None:
    for name in key_env_candidates(str(attempt.get("keyEnv", "")).strip()):
        if name and os.getenv(name):
            return name
    return None


def llm_default_display() -> str:
    attempts = llm_attempts_all()
    if not attempts:
        return "unknown"
    a = attempts[0]
    if a.get("providerId") == "legacy":
        return "agent.claw"
    return f"{a.get('providerLabel', a.get('providerId'))}/{a.get('model', '')}".strip("/")


def print_banner(metaclaw_bin: str, state: SessionState) -> None:
    sid = get_current_session_id()
    rows = [
        f"{c('Project:', '1;33')} {PROJECT_DIR}",
        f"{c('Session:', '1;33')} {sid}",
        f"{c('Agent:', '1;33')}   {AGENT_FILE.name}",
        f"{c('Runtime:', '1;33')} {get_runtime_target()}",
        f"{c('Network:', '1;33')} {network_mode_label(state.network_mode)}",
        f"{c('Vault:', '1;33')}   {vault_access_label(state.vault_access)}",
        f"{c('Scope:', '1;33')}   {state.retrieval_scope}",
        f"{c('Confirm:', '1;33')} {confirm_mode_label(state.write_confirm_mode)}",
        f"{c('Render:', '1;33')}  {state.render_mode}" + (" (glow ready)" if has_glow() else ""),
        f"{c('Focus:', '1;33')}   {focus_label(state.output_focus)}",
        f"{c('LLM:', '1;33')}     {llm_default_display()}",
        f"{c('LLM key:', '1;33')} {get_key_env_name()} {'(set)' if key_is_ready() else '(missing)'}",
        f"{c('Web key:', '1;33')} {get_web_key_env_name()} {'(set)' if web_key_is_ready() else '(missing)'}",
        f"{c('Host data:', '1;33')} {HOST_DATA_DIR}",
        f"{c('MetaClaw:', '1;33')} {metaclaw_bin}",
        "",
        f"{c('Commands:', '1;32')} /help /status /new /refresh /llm /history /render /focus /show /net /vault /scope /confirm /steps /calls /web /save /append /touch /reset /clear /exit",
    ]
    boxed("MetaClaw Obsidian Terminal Bot", rows)


def print_help() -> None:
    rows = [
        "Type any prompt and press Enter.",
        "",
        f"{c('/help', '1;32')}       Show this help",
        f"{c('/status', '1;32')}     Show runtime/key/model status",
        f"{c('/new', '1;32')}        Start a new session (/new <name>)",
        f"{c('/refresh', '1;32')}    Clear session context (history + cached outputs)",
        f"{c('/llm', '1;32')}        Show LLM provider/model/key status",
        f"{c('/llm setup', '1;32')}  Interactive provider/model/key setup (multi-select)",
        f"{c('/llm use', '1;32')}    Switch default provider/model",
        f"{c('/llm key', '1;32')}    Update API keys (hidden input, optional .env write)",
        f"{c('/history', '1;32')}    Show recent local conversation turns",
        f"{c('/render', '1;32')}     Show or set render mode (/render plain|glow|demo [--default])",
        f"{c('/focus', '1;32')}      Where you land after a response (/focus start|end|stay|input [--default])",
        f"{c('/show', '1;32')}       View last stored response (/show [start|end])",
        f"{c('/net', '1;32')}        Show or set network mode (/net none|out [--default])",
        f"{c('/vault', '1;32')}      Show or set vault access (/vault ro|rw [--default])",
        f"{c('/scope', '1;32')}      Show or set retrieval scope (/scope limited|all [--default])",
        f"{c('/confirm', '1;32')}    Show or set write confirm mode (/confirm once|diff|auto [--default])",
        f"{c('/steps', '1;32')}      Set tool max steps (/steps 16 [--default])",
        f"{c('/calls', '1;32')}      Set tool max calls per step (/calls 12 [--default])",
        f"{c('/web', '1;32')}        Force Tavily web lookup (/web <query>)",
        f"{c('/save', '1;32')}       Save last bot output to vault (/save <rel.md> | /save --default-dir <dir> [--default])",
        f"{c('/append', '1;32')}     Append last bot output to a vault file (/append <rel.md>)",
        f"{c('/touch', '1;32')}      Create an empty vault note (/touch <rel.md>)",
        f"{c('/reset', '1;32')}      Clear local memory (runtime/history.json)",
        f"{c('/clear', '1;32')}      Clear screen",
        f"{c('/exit', '1;32')}       Exit chat",
        "",
        "Menu keys:",
        "h/l move focus, space or Enter select, q/ESC cancel",
        "",
        "Web search usage:",
        "/web <query>  force Tavily web lookup + summarize (requires /net out + TAVILY_API_KEY)",
        "",
        "Runtime override example:",
        "RUNTIME_TARGET=auto ./chat.sh",
        "Config defaults example:",
        "/scope limited --default",
        "/confirm once --default",
        "Key env override example:",
        "LLM_KEY_ENV=OPENAI_FORMAT_API_KEY ./chat.sh",
        "TAVILY_KEY_ENV=TAVILY_API_KEY ./chat.sh",
    ]
    boxed("Help", rows)


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render_plain_response(text: str) -> None:
    w = width() - 4
    for raw_line in text.splitlines() or [""]:
        if raw_line.strip() == "":
            print("")
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=w,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if not wrapped:
            print("")
            continue
        for line in wrapped:
            print(f"  {line}")


def render_glow_to_ansi(text: str) -> str | None:
    glow_bin = shutil.which("glow")
    if glow_bin is None:
        return None

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmpf:
            tmpf.write(text + "\n")
            tmp_path = Path(tmpf.name)

        cmd = [glow_bin, "-s", os.getenv("GLOW_STYLE", "dark"), "-w", str(max(40, width() - 4)), str(tmp_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        return proc.stdout
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def wrap_plain_to_text(text: str) -> str:
    w = width() - 4
    out_lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        if raw_line.strip() == "":
            out_lines.append("")
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=w,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if not wrapped:
            out_lines.append("")
            continue
        for line in wrapped:
            out_lines.append(f"  {line}")
    return "\n".join(out_lines) + "\n"


def page_text(text: str, *, start_at_end: bool) -> bool:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    less_bin = shutil.which("less")
    if less_bin is None:
        return False

    cmd = [less_bin, "-R", "-F"]
    if start_at_end:
        cmd.append("+G")
    try:
        proc = subprocess.run(cmd, input=text, text=True)
        return proc.returncode in {0, 1}
    except Exception:
        return False


def present_bot_response(text: str, elapsed: float, state: SessionState) -> str:
    """Render a bot response using the selected focus behavior.

    Returns the render mode actually used: plain|glow.
    """
    header = f"{c('bot>', '1;34')} {c(f'({elapsed:.1f}s)', '2')}\n\n"

    focus = normalize_focus(state.output_focus)
    mode = normalize_render_mode(state.render_mode)

    if focus == "input":
        LAST_RESPONSE_MD_FILE.write_text(text + "\n", encoding="utf-8")
        # Still print the full response, but also keep a stored copy for `/show`.
        # This keeps the user flow "at the prompt" while preserving scrollback.
        print(header, end="")
        sys.stdout.flush()
        used_mode = "plain"
        if mode == "glow" and looks_like_markdown(text) and render_glow_response(text):
            used_mode = "glow"
        else:
            render_plain_response(text)
            used_mode = "plain"
        print("")
        print(c("meta> response stored (use /show start|end)", "2"))
        print("")
        return used_mode

    want_pager = focus in {"start", "end"} and has_less()

    ansi: str | None = None
    used_mode = "plain"
    if mode == "glow" and looks_like_markdown(text):
        ansi = render_glow_to_ansi(text)
        if ansi is not None:
            used_mode = "glow"
    if ansi is None:
        ansi = wrap_plain_to_text(text)
        used_mode = "plain"

    if want_pager:
        if page_text(header + ansi, start_at_end=(focus == "end")):
            return used_mode

    # Inline fallback (or focus=stay).
    print(header, end="")
    sys.stdout.flush()
    if used_mode == "glow" and mode == "glow" and looks_like_markdown(text) and render_glow_response(text):
        return "glow"
    render_plain_response(text)
    return "plain"


def show_last_response(state: SessionState, *, start_at_end: bool) -> None:
    if not LAST_RESPONSE_MD_FILE.exists():
        print(c("meta> no stored response yet", "2"))
        return
    text = LAST_RESPONSE_MD_FILE.read_text(encoding="utf-8", errors="ignore").rstrip("\n")
    mode = normalize_render_mode(state.render_mode)
    ansi: str | None = None
    if mode == "glow" and looks_like_markdown(text):
        ansi = render_glow_to_ansi(text)
    if ansi is None:
        ansi = wrap_plain_to_text(text)
    header = c("bot>", "1;34") + " " + c("(stored)", "2") + "\n\n"
    if page_text(header + ansi, start_at_end=start_at_end):
        return
    print(header, end="")
    sys.stdout.flush()
    if mode == "glow" and looks_like_markdown(text) and render_glow_response(text):
        return
    render_plain_response(text)


def render_glow_response(text: str) -> bool:
    glow_bin = shutil.which("glow")
    if glow_bin is None:
        return False

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmpf:
            tmpf.write(text + "\n")
            tmp_path = Path(tmpf.name)

        cmd = [glow_bin, "-s", os.getenv("GLOW_STYLE", "dark"), "-w", str(max(40, width() - 4)), str(tmp_path)]
        if sys.stdout.isatty():
            proc = subprocess.run(cmd, text=True)
            return proc.returncode == 0

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return False
        output = proc.stdout.rstrip("\n")
        if output:
            print(output)
        return True
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def looks_like_markdown(text: str) -> bool:
    markers = [
        r"(^|\n)#{1,6}\s",
        r"(^|\n)\s*[-*+]\s+",
        r"(^|\n)\s*\d+\.\s+",
        r"```",
        r"\[[^\]]+\]\([^)]+\)",
        r"(^|\n)>\s+",
    ]
    return any(re.search(pattern, text) for pattern in markers)


def pretty_print_bot_response(text: str, elapsed: float, render_mode: str) -> str:
    tag = c("bot>", "1;34")
    meta = c(f"({elapsed:.1f}s)", "2")
    print(f"{tag} {meta}")
    print("")
    sys.stdout.flush()

    mode = normalize_render_mode(render_mode)
    if mode == "glow" and looks_like_markdown(text) and render_glow_response(text):
        return "glow"

    render_plain_response(text)
    return "plain"


def tail_log(path: Path, n: int = 30) -> list[str]:
    if not path.exists():
        return ["(no log file)"]
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n:]


def spinner_wait(proc: subprocess.Popen, label: str = "running") -> None:
    frames = "|/-\\"
    i = 0
    while proc.poll() is None:
        frame = frames[i % len(frames)]
        sys.stdout.write(f"\r{c('meta>', '1;35')} {label} {frame}")
        sys.stdout.flush()
        time.sleep(0.09)
        i += 1
    sys.stdout.write("\r" + " " * (len(label) + 14) + "\r")
    sys.stdout.flush()


def rewrite_llm_block(agent_text: str, attempt: dict) -> str:
    """Best-effort YAML rewrite for agent.llm fields (no YAML parser dependency)."""
    engine_provider = str(attempt.get("engineProvider", "")).strip()
    model = str(attempt.get("model", "")).strip()
    base_url = str(attempt.get("baseURL", "")).strip()
    if not engine_provider or not model:
        return agent_text

    # Container-side key env depends on provider protocol.
    container_key_env = "OPENAI_API_KEY"
    if engine_provider == "anthropic":
        container_key_env = "ANTHROPIC_API_KEY"

    lines = agent_text.splitlines()
    in_llm = False
    llm_indent = 0
    seen = {"provider": False, "model": False, "baseURL": False, "apiKeyEnv": False}
    llm_line_idx: int | None = None

    def indent_len(s: str) -> int:
        return len(s) - len(s.lstrip(" \t"))

    for i, line in enumerate(lines):
        trimmed = line.strip()
        indent = indent_len(line)
        if trimmed == "llm:":
            in_llm = True
            llm_indent = indent
            llm_line_idx = i
            continue
        if in_llm and indent <= llm_indent and trimmed and not trimmed.startswith("#"):
            in_llm = False
        if not in_llm:
            continue

        key_match = re.match(r"\s*([A-Za-z0-9_]+):\s*(.*)$", line)
        if not key_match:
            continue
        key = key_match.group(1)
        prefix = line[:indent]
        if key == "provider":
            lines[i] = prefix + f"provider: {engine_provider}"
            seen["provider"] = True
        elif key == "model":
            lines[i] = prefix + f"model: {model}"
            seen["model"] = True
        elif key == "baseURL":
            if base_url:
                lines[i] = prefix + f"baseURL: {base_url}"
                seen["baseURL"] = True
        elif key == "apiKeyEnv":
            # Keep secrets in host env; container always reads a provider-appropriate env name.
            lines[i] = prefix + f"apiKeyEnv: {container_key_env}"
            seen["apiKeyEnv"] = True

    if llm_line_idx is None:
        return agent_text

    # Insert missing keys right after `llm:` line, in a stable order.
    insert_at = llm_line_idx + 1
    indent = " " * (llm_indent + 2)
    inserts: list[str] = []
    if not seen["provider"]:
        inserts.append(indent + f"provider: {engine_provider}")
    if not seen["model"]:
        inserts.append(indent + f"model: {model}")
    if base_url and not seen["baseURL"]:
        inserts.append(indent + f"baseURL: {base_url}")
    if not seen["apiKeyEnv"]:
        inserts.append(indent + f"apiKeyEnv: {container_key_env}")
    if inserts:
        lines = lines[:insert_at] + inserts + lines[insert_at:]

    return "\n".join(lines)


def set_habitat_env_kv(agent_text: str, key: str, value: str) -> str:
    """Best-effort YAML rewrite for agent.habitat.env KEY: VALUE (no YAML parser dependency)."""
    lines = agent_text.splitlines()
    in_habitat = False
    in_env = False
    habitat_indent = 0
    env_indent = 0
    env_line_idx: int | None = None
    found_key = False

    def indent_len(s: str) -> int:
        return len(s) - len(s.lstrip(" \t"))

    for i, line in enumerate(lines):
        trimmed = line.strip()
        indent = indent_len(line)

        if trimmed == "habitat:":
            in_habitat = True
            habitat_indent = indent
            in_env = False
            continue
        if in_habitat and indent <= habitat_indent and trimmed and not trimmed.startswith("#"):
            in_habitat = False
            in_env = False

        if in_habitat and trimmed == "env:":
            in_env = True
            env_indent = indent
            env_line_idx = i
            continue
        if in_env and indent <= env_indent and trimmed and not trimmed.startswith("#"):
            in_env = False

        if not in_env:
            continue

        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if not m:
            continue
        if m.group(1) == key:
            prefix = line[:indent]
            lines[i] = prefix + f"{key}: {value}"
            found_key = True
            break

    if env_line_idx is None:
        return agent_text

    if not found_key:
        insert_at = env_line_idx + 1
        indent = " " * (env_indent + 2)
        lines = lines[:insert_at] + [indent + f"{key}: {value}"] + lines[insert_at:]
    return "\n".join(lines)


def write_effective_agent(
    network_mode: str,
    vault_access: str,
    *,
    llm_attempt: dict | None,
    tool_max_steps: int,
    tool_max_calls_per_step: int,
) -> Path:
    raw = AGENT_FILE.read_text(encoding="utf-8")
    if network_mode == "none":
        updated = raw.replace("mode: outbound", "mode: none", 1)
    else:
        updated = raw.replace("mode: none", "mode: outbound", 1)

    if updated == raw and f"mode: {network_mode}" not in raw:
        raise RuntimeError("cannot resolve habitat.network.mode in agent.claw")

    updated = set_mount_readonly_by_target(updated, "/vault", normalize_vault_access(vault_access) == "ro")
    if llm_attempt is not None and llm_attempt.get("providerId") != "legacy":
        updated = rewrite_llm_block(updated, llm_attempt)
    # Tool-loop budgets are container-side env vars.
    updated = set_habitat_env_kv(updated, "BOT_TOOL_MAX_STEPS", f"\"{normalize_tool_max_steps(tool_max_steps)}\"")
    updated = set_habitat_env_kv(
        updated,
        "BOT_TOOL_MAX_CALLS_PER_STEP",
        f"\"{normalize_tool_max_calls_per_step(tool_max_calls_per_step)}\"",
    )

    EFFECTIVE_AGENT_FILE.write_text(updated, encoding="utf-8")
    return EFFECTIVE_AGENT_FILE


def write_session_config(state: SessionState) -> None:
    payload = {
        "retrieval_scope": state.retrieval_scope,
    }
    SESSION_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_transient_bot_error(text: str) -> bool:
    lower = (text or "").lower()
    patterns = [
        "http 408",
        "http 425",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "remote end closed connection",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "unavailable",
        "overloaded",
    ]
    return any(p in lower for p in patterns)


def metaclaw_run(metaclaw_bin: str, prompt: str, state: SessionState) -> tuple[bool, float]:
    REQUEST_FILE.write_text(prompt, encoding="utf-8")
    write_session_config(state)
    # Truncate response without unlinking (paths may be stable session symlinks).
    try:
        RESPONSE_FILE.write_text("", encoding="utf-8")
    except Exception:
        pass

    # Keep agent context (AGENTS.md + soul.md) synced into the mounted runtime dir.
    try:
        write_agents_context()
    except Exception:
        pass

    runtime_target = get_runtime_target()
    attempts_all = llm_attempts_all()
    if not attempts_all:
        RESPONSE_FILE.write_text("[meta error] no LLM attempts configured\n", encoding="utf-8")
        return False, 0.0

    # Never silently "switch providers" because a key is missing. If the default
    # provider/model is selected, require a usable key for that attempt.
    default_attempt = attempts_all[0]
    if pick_first_set_env_key(default_attempt) is None:
        needed = ", ".join(key_env_candidates(str(default_attempt.get("keyEnv", "")).strip()))
        RESPONSE_FILE.write_text(
            "[meta error] LLM key is missing.\n"
            f"Set one of: {needed}\n"
            "Or run: /llm key\n",
            encoding="utf-8",
        )
        return False, 0.0

    # Default first; fallbacks must also have a usable key.
    attempts = [default_attempt] + [a for a in attempts_all[1:] if pick_first_set_env_key(a) is not None]

    start = time.time()
    last_err_text = ""
    for i, attempt in enumerate(attempts):
        key_env = pick_first_set_env_key(attempt) or get_key_env_name()
        llm_override = attempt if LLM_CONFIG_FILE.exists() and attempt.get("providerId") != "legacy" else None
        agent_file = write_effective_agent(
            state.network_mode,
            state.vault_access,
            llm_attempt=llm_override,
            tool_max_steps=state.tool_max_steps,
            tool_max_calls_per_step=state.tool_max_calls_per_step,
        )

        run_args = [
            metaclaw_bin,
            "run",
            str(agent_file),
            f"--state-dir={STATE_DIR}",
            f"--llm-api-key-env={key_env}",
        ]
        web_key_env = get_web_key_env_name()
        if os.getenv(web_key_env):
            run_args.append(f"--secret-env={web_key_env}")
        if runtime_target != "auto":
            run_args.append(f"--runtime={runtime_target}")

        if i > 0:
            label = attempt.get("providerLabel", attempt.get("providerId", "llm"))
            model = attempt.get("model", "")
            print(c(f"meta> retrying with fallback: {label}/{model}", "33"))

        with RUN_LOG_FILE.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                run_args,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_DIR),
                env=os.environ.copy(),
                text=True,
            )
            spinner_wait(proc, label=f"{runtime_target}/{network_mode_label(state.network_mode)} executing")
            rc = proc.wait()

        if rc == 0:
            return True, time.time() - start

        err_text = read_response_text()
        last_err_text = err_text
        if i < len(attempts) - 1 and err_text.startswith("[bot error]") and is_transient_bot_error(err_text):
            continue
        break

    # Failure
    if last_err_text:
        try:
            RESPONSE_FILE.write_text(last_err_text + "\n", encoding="utf-8")
        except Exception:
            pass
    return False, time.time() - start


def local_status(state: SessionState) -> None:
    sid = get_current_session_id()
    rows = [
        f"{c('Session:', '1;33')} {sid}",
        f"{c('Runtime target:', '1;33')} {get_runtime_target()}",
        f"{c('Network mode:', '1;33')} {network_mode_label(state.network_mode)}",
        f"{c('Vault access:', '1;33')} {vault_access_label(state.vault_access)}",
        f"{c('Scope mode:', '1;33')} {state.retrieval_scope}",
        f"{c('Confirm mode:', '1;33')} {confirm_mode_label(state.write_confirm_mode)}",
        f"{c('Render mode:', '1;33')} {state.render_mode}" + (" (glow ready)" if has_glow() else " (glow missing)"),
        f"{c('Focus mode:', '1;33')} {focus_label(state.output_focus)}",
        f"{c('Tool budget:', '1;33')} steps={state.tool_max_steps} calls/step={state.tool_max_calls_per_step}",
        f"{c('Save default dir:', '1;33')} {state.save_default_dir}",
        f"{c('LLM default:', '1;33')} {llm_default_display()}",
        f"{c('LLM key env:', '1;33')} {get_key_env_name()} {'(set)' if key_is_ready() else '(missing)'}",
        f"{c('Web key env:', '1;33')} {get_web_key_env_name()} {'(set)' if web_key_is_ready() else '(missing)'}",
        f"{c('Host data dir:', '1;33')} {HOST_DATA_DIR}",
        f"{c('Host workspace dir:', '1;33')} {HOST_WORKSPACE_DIR}",
        f"{c('Host config dir:', '1;33')} {HOST_CONFIG_DIR}",
        f"{c('State dir:', '1;33')} {STATE_DIR}",
        f"{c('Runtime dir:', '1;33')} {RUNTIME_DIR}",
        f"{c('History file:', '1;33')} {HISTORY_FILE} {'(exists)' if HISTORY_FILE.exists() else '(not found)'}",
    ]
    boxed("Status", rows)


def show_local_history() -> None:
    if not HISTORY_FILE.exists():
        print(c("meta> no local history yet", "2"))
        return

    try:
        records = json.loads(HISTORY_FILE.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        print(c("meta> history file is not valid JSON", "31"))
        return

    if not isinstance(records, list) or not records:
        print(c("meta> history is empty", "2"))
        return

    rows: list[str] = []
    for item in records[-8:]:
        role = str(item.get("role", "?")) if isinstance(item, dict) else "?"
        content = str(item.get("content", "")) if isinstance(item, dict) else str(item)
        content = content.replace("\n", " ").strip()
        content = content[:120] + ("..." if len(content) > 120 else "")
        rows.append(f"{c(role + ':', '1;34')} {content}")

    boxed("Recent History", rows)


def reset_memory() -> None:
    # Do not unlink runtime paths, because they may be stable session symlinks.
    try:
        HISTORY_FILE.write_text("[]\n", encoding="utf-8")
    except Exception:
        pass
    try:
        RESPONSE_FILE.write_text("", encoding="utf-8")
    except Exception:
        pass
    print(c("meta> memory reset (history cleared)", "1;32"))


def refresh_context() -> None:
    # A stronger reset for the current session: clears history, cached response, and session state.
    try:
        HISTORY_FILE.write_text("[]\n", encoding="utf-8")
    except Exception:
        pass
    for p in (RESPONSE_FILE, REQUEST_FILE, SESSION_FILE, LAST_RESPONSE_MD_FILE, AGENTS_CONTEXT_FILE):
        try:
            p.write_text("", encoding="utf-8")
        except Exception:
            pass
    print(c("meta> session refreshed (context cleared)", "1;32"))


def read_response_text() -> str:
    if not RESPONSE_FILE.exists():
        return "[bot error] response file not generated"
    text = RESPONSE_FILE.read_text(encoding="utf-8", errors="ignore").rstrip("\n")
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text


def prompt_uses_llm(prompt: str) -> bool:
    # Any non-TUI command that reaches the runtime will require an LLM call.
    return True


def check_key_env_or_warn(prompt: str) -> bool:
    if LLM_CONFIG_FILE.exists():
        ready = llm_attempts_ready()
        if not ready:
            cfg = load_llm_config()
            envs = []
            for p in cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []:
                if isinstance(p, dict):
                    k = str(p.get("keyEnv", "")).strip()
                    if k:
                        envs.append(k)
            envs = sorted(set(envs))
            print(c("meta> missing LLM API key(s) for selected providers", "31"))
            if envs:
                print(c("meta> set one of these env vars (or run /llm key):", "2"))
                for k in envs:
                    print(c(f"  export {k}=your_key", "2"))
            else:
                print(c("meta> run /llm setup to configure providers/models", "2"))
            return False
        # If default key is missing but another provider is available, warn but proceed.
        default_env = get_key_env_name()
        if default_env and not os.getenv(default_env):
            print(c(f"meta> WARNING: default LLM key env {default_env} is missing; will use a fallback provider/model", "33"))
    else:
        key_env = get_key_env_name()
        if not os.getenv(key_env):
            print(c(f"meta> missing LLM API key env: {key_env}", "31"))
            print(c(f"meta> example: export {key_env}=your_key", "2"))
            print(c("meta> tip: run /llm setup for multi-provider config", "2"))
            return False

    web_key_env = get_web_key_env_name()
    if prompt.startswith("/web ") and not os.getenv(web_key_env):
        print(c(f"meta> web search key env {web_key_env} is missing (web lookup disabled)", "33"))
        print(c(f"meta> set it to enable Tavily: export {web_key_env}=your_key", "2"))
    return True


def render_mode_demo(render_mode: str) -> str:
    sample = (
        "# Markdown Preview\n\n"
        "- This is a bullet item.\n"
        "- This is **bold** text.\n"
        "- This is `inline code`.\n\n"
        "```python\n"
        "print('hello from glow')\n"
        "```\n\n"
        "[Example Link](https://example.com)\n"
    )
    return pretty_print_bot_response(sample, 0.0, render_mode)


def parse_command_tokens(text: str) -> list[str]:
    # /web frequently includes apostrophes (today's) which should NOT behave like shell quoting.
    # Handle it as a raw command early to avoid shlex errors.
    if text.startswith("/web "):
        return ["/web", text[len("/web ") :].strip()]
    try:
        return shlex.split(text)
    except Exception as exc:
        # Fallback: keep the first token as the command, and treat the rest as raw text.
        # This makes commands resilient to unbalanced quotes in user prompts.
        stripped = (text or "").strip()
        if not stripped.startswith("/"):
            print(c(f"meta> command parse error: {exc}", "31"))
            return []
        parts = stripped.split(maxsplit=1)
        if not parts:
            print(c(f"meta> command parse error: {exc}", "31"))
            return []
        if len(parts) == 1:
            return [parts[0]]
        return [parts[0], parts[1]]


def parse_value_and_default(tokens: list[str]) -> tuple[str | None, bool]:
    args = list(tokens)
    persist = False
    out = []
    for token in args:
        if token == "--default":
            persist = True
        else:
            out.append(token)
    value = out[0] if out else None
    return value, persist


def prompt_int(title: str, *, current: int, min_value: int, max_value: int) -> int | None:
    raw = input(c(f"{title} [{current}]: ", "2")).strip()
    if raw == "":
        return current
    try:
        n = int(raw)
    except Exception:
        print(c("meta> invalid number", "31"))
        return None
    if n < min_value or n > max_value:
        print(c(f"meta> out of range ({min_value}-{max_value})", "31"))
        return None
    return n


def prompt_select(title: str, choices: list[tuple[str, str]], current_value: str) -> str | None:
    values = [v for v, _ in choices]
    if current_value in values:
        idx = values.index(current_value)
    else:
        idx = len(choices) // 2

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        labels = ", ".join(f"{label}={value}" for value, label in choices)
        print(c(f"meta> {title} options: {labels}", "2"))
        print(c("meta> non-interactive terminal; pass explicit value", "2"))
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    hint = "(h/l move, space/enter select, q cancel)"

    try:
        tty.setraw(fd)
        while True:
            labels = []
            for i, (value, label) in enumerate(choices):
                text = f"{label}"
                if i == idx:
                    labels.append(c(f"[{text}]", "1;32"))
                else:
                    labels.append(f" {text} ")
            line = f"\r{c('meta>', '1;35')} {title}: " + "  ".join(labels) + f"  {c(hint, '2')}"
            sys.stdout.write("\r" + " " * (width() - 1))
            sys.stdout.write(line)
            sys.stdout.flush()

            ch = sys.stdin.read(1)
            if ch in {"h", "H"}:
                idx = (idx - 1) % len(choices)
                continue
            if ch in {"l", "L"}:
                idx = (idx + 1) % len(choices)
                continue
            if ch in {" ", "\r", "\n"}:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return choices[idx][0]
            if ch in {"q", "Q", "\x03"}:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return None
            if ch == "\x1b":
                nxt = sys.stdin.read(1)
                if nxt != "[":
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return None
                arrow = sys.stdin.read(1)
                if arrow == "D":
                    idx = (idx - 1) % len(choices)
                    continue
                if arrow == "C":
                    idx = (idx + 1) % len(choices)
                    continue
                sys.stdout.write("\n")
                sys.stdout.flush()
                return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def prompt_menu(title: str, choices: list[tuple[str, str]], current_value: str | None = None) -> str | None:
    """Vertical single-select menu (arrow keys + Enter)."""
    if not choices:
        return None
    values = [v for v, _ in choices]
    idx = 0
    if current_value and current_value in values:
        idx = values.index(current_value)

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        labels = ", ".join(f"{label}={value}" for value, label in choices)
        print(c(f"meta> {title} options: {labels}", "2"))
        print(c("meta> non-interactive terminal; pass explicit value", "2"))
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    hint = "(use ↑/↓, Enter; q cancel)"
    lines = len(choices) + 1

    def crlf() -> None:
        sys.stdout.write("\r\n")

    def print_line(s: str) -> None:
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.write(s)
        crlf()

    def render() -> None:
        print_line(f"{c('meta>', '1;35')} {title} {c(hint, '2')}")
        for i, (value, label) in enumerate(choices):
            prefix = "> " if i == idx else "  "
            text = label
            if i == idx:
                text = c(label, "1;32")
            print_line(prefix + text)

    def clear_menu() -> None:
        # Move cursor to the top of the menu and clear below.
        sys.stdout.write(f"\x1b[{lines}A\r\x1b[J")

    def redraw() -> None:
        clear_menu()
        render()
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        render()
        sys.stdout.flush()

        while True:
            ch = sys.stdin.read(1)
            if ch in {"q", "Q", "\x03"}:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return None
            if ch in {"\r", "\n", " "}:
                clear_menu()
                sys.stdout.write("\x1b[?25h")
                sys.stdout.flush()
                return choices[idx][0]
            if ch in {"k", "K"}:
                idx = (idx - 1) % len(choices)
                redraw()
                continue
            if ch in {"j", "J"}:
                idx = (idx + 1) % len(choices)
                redraw()
                continue
            if ch == "\x1b":
                nxt = sys.stdin.read(1)
                if nxt != "[":
                    continue
                arrow = sys.stdin.read(1)
                if arrow == "A":  # up
                    idx = (idx - 1) % len(choices)
                    redraw()
                    continue
                if arrow == "B":  # down
                    idx = (idx + 1) % len(choices)
                    redraw()
                    continue
                continue
    finally:
        try:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
        except Exception:
            pass
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def prompt_multiselect(title: str, choices: list[tuple[str, str]], initial: list[str] | None = None) -> list[str] | None:
    """Vertical multi-select menu (arrow keys, space toggles, Enter confirms)."""
    if not choices:
        return None
    selected: set[str] = set(str(x).strip() for x in (initial or []) if str(x).strip())
    idx = 0

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        labels = ", ".join(f"{label}={value}" for value, label in choices)
        print(c(f"meta> {title} options: {labels}", "2"))
        print(c("meta> non-interactive terminal; pass explicit value", "2"))
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    hint = "(use ↑/↓, space toggle, Enter done; q cancel)"
    lines = len(choices) + 1

    def crlf() -> None:
        sys.stdout.write("\r\n")

    def print_line(s: str) -> None:
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.write(s)
        crlf()

    def render() -> None:
        print_line(f"{c('meta>', '1;35')} {title} {c(hint, '2')}")
        for i, (value, label) in enumerate(choices):
            checked = value in selected
            box = "[x]" if checked else "[ ]"
            prefix = "> " if i == idx else "  "
            line = f"{box} {label}"
            if i == idx:
                line = c(line, "1;32")
            print_line(prefix + line)

    def clear_menu() -> None:
        sys.stdout.write(f"\x1b[{lines}A\r\x1b[J")

    def redraw() -> None:
        clear_menu()
        render()
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        render()
        sys.stdout.flush()

        while True:
            ch = sys.stdin.read(1)
            if ch in {"q", "Q", "\x03"}:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return None
            if ch in {"\r", "\n"}:
                clear_menu()
                sys.stdout.write("\x1b[?25h")
                sys.stdout.flush()
                # Stable order: follow choices order.
                out = [value for value, _ in choices if value in selected]
                return out
            if ch == " ":
                value = choices[idx][0]
                if value in selected:
                    selected.remove(value)
                else:
                    selected.add(value)
                redraw()
                continue
            if ch in {"k", "K"}:
                idx = (idx - 1) % len(choices)
                redraw()
                continue
            if ch in {"j", "J"}:
                idx = (idx + 1) % len(choices)
                redraw()
                continue
            if ch == "\x1b":
                nxt = sys.stdin.read(1)
                if nxt != "[":
                    continue
                arrow = sys.stdin.read(1)
                if arrow == "A":
                    idx = (idx - 1) % len(choices)
                    redraw()
                    continue
                if arrow == "B":
                    idx = (idx + 1) % len(choices)
                    redraw()
                    continue
                continue
    finally:
        try:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
        except Exception:
            pass
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def apply_mode_change(
    state: SessionState,
    defaults: dict[str, str],
    key: str,
    value: str,
    persist: bool,
) -> None:
    if key == "network_mode":
        state.network_mode = normalize_network_mode(value)
        display = network_mode_label(state.network_mode)
    elif key == "render_mode":
        state.render_mode = normalize_render_mode(value)
        display = state.render_mode
    elif key == "vault_access":
        state.vault_access = normalize_vault_access(value)
        display = vault_access_label(state.vault_access)
    elif key == "retrieval_scope":
        state.retrieval_scope = normalize_scope(value)
        display = state.retrieval_scope
    elif key == "write_confirm_mode":
        state.write_confirm_mode = normalize_confirm_mode(value)
        display = confirm_mode_label(state.write_confirm_mode)
    elif key == "output_focus":
        state.output_focus = normalize_focus(value)
        display = focus_label(state.output_focus)
    else:
        raise ValueError(f"unsupported mode key: {key}")

    print(c(f"meta> {key} set to {display}", "1;32"))
    if persist:
        defaults[key] = value
        defaults["network_mode"] = normalize_network_mode(defaults["network_mode"])
        defaults["render_mode"] = normalize_render_mode(defaults["render_mode"])
        defaults["vault_access"] = normalize_vault_access(defaults.get("vault_access", DEFAULT_VALUES["vault_access"]))
        defaults["retrieval_scope"] = normalize_scope(defaults["retrieval_scope"])
        defaults["write_confirm_mode"] = normalize_confirm_mode(defaults["write_confirm_mode"])
        defaults["output_focus"] = normalize_focus(defaults.get("output_focus", DEFAULT_VALUES["output_focus"]))
        save_defaults(defaults)
        print(c("meta> saved as project default", "2"))


def llm_status_rows() -> list[str]:
    if not LLM_CONFIG_FILE.exists():
        return [
            f"{c('mode:', '1;33')} legacy (no llm.config.json yet)",
            f"{c('key env:', '1;33')} {os.getenv('LLM_KEY_ENV', 'OPENAI_FORMAT_API_KEY')} ({'set' if key_is_ready() else 'missing'})",
            "",
            "Tip: run `/llm setup` to pick providers/models and store a default.",
        ]

    cfg = load_llm_config()
    default = cfg.get("default") if isinstance(cfg.get("default"), dict) else {}
    default_pid = str(default.get("providerId", "")).strip()
    default_model = str(default.get("model", "")).strip()
    rows: list[str] = []
    rows.append(f"{c('default:', '1;33')} {llm_default_display()}")
    rows.append("")
    providers = cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []
    for p in providers:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", "")).strip()
        label = str(p.get("label", "")).strip() or pid
        key_env = str(p.get("keyEnv", "")).strip()
        key_state = "set" if (key_env and os.getenv(key_env)) else "missing"
        selected = p.get("selectedModels") if isinstance(p.get("selectedModels"), list) else []
        selected_models = [str(m).strip() for m in selected if str(m).strip()]
        model_list = ", ".join(selected_models) if selected_models else "(none)"
        default_mark = ""
        if pid == default_pid:
            default_mark = c("  [default provider]", "2")
        rows.append(f"{c(label+':', '1;33')} {model_list}{default_mark}")
        if key_env:
            rows.append(f"  key env: {key_env} ({key_state})")
        if pid == default_pid and default_model:
            rows.append(f"  default model: {default_model}")
        rows.append("")
    if rows and rows[-1] == "":
        rows.pop()
    return rows


def llm_show_status() -> None:
    boxed("LLM", llm_status_rows())


def llm_collect_keys(cfg: dict, provider_ids: list[str]) -> None:
    updates: dict[str, str] = {}
    providers = cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []
    id_set = set(provider_ids)
    for p in providers:
        if not isinstance(p, dict):
            continue
        if str(p.get("id", "")).strip() not in id_set:
            continue
        label = str(p.get("label", "")).strip() or str(p.get("id", "")).strip()
        key_env = str(p.get("keyEnv", "")).strip()
        if not key_env:
            continue
        prompt = f"Enter {key_env} for {label} (hidden; leave empty to skip): "
        try:
            value = getpass.getpass(prompt)
        except Exception:
            value = ""
        value = (value or "").strip()
        if not value:
            continue
        os.environ[key_env] = value
        updates[key_env] = value

    if not updates:
        print(c("meta> no keys updated", "2"))
        return

    save = prompt_menu("Save entered keys into .env (gitignored)?", [("yes", "yes"), ("no", "no")], "yes")
    if save == "yes":
        try:
            dotenv_upsert(DOTENV_FILE, updates)
            print(c("meta> keys saved to .env", "2"))
        except Exception as exc:
            print(c(f"meta> failed to write .env: {exc}", "33"))
    else:
        print(c("meta> keys kept in current process env only", "2"))


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,64}$")


def is_valid_env_name(name: str) -> bool:
    return bool(_ENV_NAME_RE.match((name or "").strip()))


def sanitize_provider_id(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        return ""
    if not s[0].isalpha():
        s = "p-" + s
    s = s[:64]
    return s


def suggested_key_env_for_provider(pid: str) -> str:
    if pid in LLM_PROVIDER_CATALOG:
        return str(LLM_PROVIDER_CATALOG[pid].get("keyEnv", "")).strip()
    base = re.sub(r"[^A-Za-z0-9]+", "_", (pid or "custom")).upper().strip("_") or "CUSTOM"
    return f"{base}_API_KEY"


def llm_provider_choices(cfg: dict) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    # Built-ins first (stable order).
    for pid in LLM_PROVIDER_CATALOG.keys():
        choices.append((pid, str(LLM_PROVIDER_CATALOG[pid].get("label", pid)).strip() or pid))
    # Then any custom providers already present in cfg.
    providers = cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []
    custom: list[tuple[str, str]] = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", "")).strip()
        if not pid or pid in LLM_PROVIDER_CATALOG:
            continue
        label = str(p.get("label", "")).strip() or pid
        custom.append((pid, f"{label} (custom)"))
    # stable ordering for customs
    custom.sort(key=lambda x: x[0])
    choices.extend(custom)
    return choices


def llm_add_custom_provider(cfg: dict) -> dict:
    cfg = cfg if isinstance(cfg, dict) else default_llm_config()
    providers = cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []

    print(c("meta> add custom provider (OpenAI-compatible or Anthropic)", "2"))

    label = input(c("Provider name (label): ", "2")).strip()
    if not label:
        print(c("meta> cancelled (empty label)", "33"))
        return cfg
    pid_default = sanitize_provider_id(label)
    pid = input(c(f"Provider id [{pid_default}]: ", "2")).strip() or pid_default
    pid = sanitize_provider_id(pid)
    if not pid or not _PROVIDER_ID_RE.match(pid):
        print(c("meta> invalid provider id (use lowercase letters, numbers, '-' '_')", "31"))
        return cfg

    protocol = prompt_menu(
        "Provider protocol",
        [
            ("openai_compatible", "OpenAI-compatible (chat/completions)"),
            ("anthropic", "Anthropic (Claude) (v1/messages)"),
        ],
        "openai_compatible",
    )
    if protocol is None:
        return cfg

    base_url = input(c("Base URL (e.g. https://api.openai.com/v1): ", "2")).strip()
    if not base_url:
        print(c("meta> cancelled (empty base URL)", "33"))
        return cfg

    key_env_default = suggested_key_env_for_provider(pid)
    key_env = input(c(f"API key env name [{key_env_default}]: ", "2")).strip() or key_env_default
    if not is_valid_env_name(key_env):
        print(c("meta> invalid env var name", "31"))
        return cfg

    models_raw = input(c("Models (comma separated; optional): ", "2")).strip()
    models = [m.strip() for m in models_raw.split(",") if m.strip()] if models_raw else []
    if not models:
        models = ["gpt-4o-mini"] if protocol == "openai_compatible" else ["claude-3-5-sonnet-latest"]
    selected_models = [models[0]]

    # Upsert provider in cfg.
    new_entry = {
        "id": pid,
        "label": label,
        "engineProvider": protocol,
        "baseURL": base_url,
        "keyEnv": key_env,
        "models": models,
        "selectedModels": selected_models,
    }
    replaced = False
    for i, p in enumerate(providers):
        if isinstance(p, dict) and str(p.get("id", "")).strip() == pid:
            providers[i] = new_entry
            replaced = True
            break
    if not replaced:
        providers.append(new_entry)
    cfg["providers"] = providers
    cfg = normalize_llm_config(cfg)

    # Collect key (optional) right away.
    try:
        value = getpass.getpass(f"Enter {key_env} value (hidden; leave empty to skip): ")
    except Exception:
        value = ""
    value = (value or "").strip()
    if value:
        os.environ[key_env] = value
        save = prompt_menu("Save this key into .env (gitignored)?", [("yes", "yes"), ("no", "no")], "yes")
        if save == "yes":
            try:
                dotenv_upsert(DOTENV_FILE, {key_env: value})
                print(c("meta> key saved to .env", "2"))
            except Exception as exc:
                print(c(f"meta> failed to write .env: {exc}", "33"))
        else:
            print(c("meta> key kept in current process env only", "2"))

    print(c(f"meta> added provider: {label} ({pid})", "1;32"))
    return cfg


def llm_delete_custom_providers(cfg: dict) -> dict:
    cfg = cfg if isinstance(cfg, dict) else default_llm_config()
    providers = cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []
    custom = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", "")).strip()
        if pid and pid not in LLM_PROVIDER_CATALOG:
            label = str(p.get("label", "")).strip() or pid
            custom.append((pid, f"{label} (custom)"))
    if not custom:
        print(c("meta> no custom providers to delete", "2"))
        return cfg
    to_del = prompt_multiselect("Delete custom providers", custom, [])
    if to_del is None:
        return cfg
    if not to_del:
        print(c("meta> nothing selected", "2"))
        return cfg
    cfg["providers"] = [p for p in providers if not (isinstance(p, dict) and str(p.get("id", "")).strip() in set(to_del))]
    cfg = normalize_llm_config(cfg)
    print(c(f"meta> deleted: {', '.join(to_del)}", "1;32"))
    return cfg


def llm_setup_interactive() -> None:
    # Optional: manage custom providers first.
    existing_cfg = load_llm_config() if LLM_CONFIG_FILE.exists() else default_llm_config()
    while True:
        action = prompt_menu(
            "LLM setup",
            [
                ("continue", "continue"),
                ("add_custom", "add custom provider"),
                ("del_custom", "delete custom provider"),
            ],
            "continue",
        )
        if action is None:
            return
        if action == "add_custom":
            existing_cfg = llm_add_custom_provider(existing_cfg)
            continue
        if action == "del_custom":
            existing_cfg = llm_delete_custom_providers(existing_cfg)
            continue
        break

    # Step 0: choose whether to edit existing selection or start fresh.
    # A fresh start avoids the common confusion where Gemini is pre-selected.
    mode = prompt_menu(
        "LLM setup mode",
        [
            ("fresh", "start fresh (no pre-selected providers)"),
            ("edit", "edit existing selection"),
        ],
        "fresh",
    )
    if mode is None:
        return

    # Step 1: providers (multi-select)
    existing_ids = [str(p.get("id", "")).strip() for p in existing_cfg.get("providers", []) if isinstance(p, dict)]
    initial_ids: list[str] = existing_ids if (mode == "edit" and LLM_CONFIG_FILE.exists()) else []
    provider_choices = llm_provider_choices(existing_cfg)
    selected_ids = prompt_multiselect("LLM providers", provider_choices, initial_ids)
    if selected_ids is None:
        return
    if not selected_ids:
        print(c("meta> no providers selected", "33"))
        return
    selected_labels: list[str] = []
    for pid in selected_ids:
        prev = next((p for p in existing_cfg.get("providers", []) if isinstance(p, dict) and str(p.get("id", "")).strip() == pid), None)
        cat = LLM_PROVIDER_CATALOG.get(pid, {})
        label = (str((prev or {}).get("label", "")).strip() or str(cat.get("label", pid)).strip() or pid)
        selected_labels.append(label)
    print(c(f"meta> selected providers: {', '.join(selected_labels)}", "2"))

    # Step 2: models per provider (multi-select).
    # Note: If you selected multiple providers, you'll configure them one-by-one.
    new_providers: list[dict] = []
    for i, pid in enumerate(selected_ids):
        cat = LLM_PROVIDER_CATALOG.get(pid, {})
        prev = next((p for p in existing_cfg.get("providers", []) if isinstance(p, dict) and str(p.get("id", "")).strip() == pid), None)
        label = (str((prev or {}).get("label", "")).strip() or str(cat.get("label", pid)).strip() or pid)
        models = []
        if isinstance(prev, dict) and isinstance(prev.get("models"), list):
            models.extend([str(m).strip() for m in prev["models"] if str(m).strip()])
        models.extend([str(m).strip() for m in (cat.get("models") or []) if str(m).strip()])
        # de-dupe order
        seen = set()
        models = [m for m in models if not (m in seen or seen.add(m))]
        if not models:
            models = ["gpt-4o-mini"]

        prev_sel = []
        if isinstance(prev, dict) and isinstance(prev.get("selectedModels"), list):
            prev_sel = [str(m).strip() for m in prev["selectedModels"] if str(m).strip()]
        initial_sel = [m for m in prev_sel if m in models] or models[:1]

        model_choices = [(m, m) for m in models]
        if len(selected_ids) > 1:
            title = f"Models for {label} ({i+1}/{len(selected_ids)})"
        else:
            title = f"Models for {label}"
        selected_models = prompt_multiselect(title, model_choices, initial_sel)
        if selected_models is None:
            return
        if not selected_models:
            print(c("meta> you must select at least one model", "33"))
            return

        # Optional custom additions.
        extra = input(c(f"meta> add custom models for {label} (comma separated, optional): ", "2")).strip()
        extra_models = [x.strip() for x in extra.split(",") if x.strip()] if extra else []
        for m in extra_models:
            if m not in models:
                models.append(m)
            if m not in selected_models:
                selected_models.append(m)

        new_providers.append(
            {
                "id": pid,
                "label": label,
                "engineProvider": (str((prev or {}).get("engineProvider", "")).strip() or str(cat.get("engineProvider", "")).strip()),
                "baseURL": (str((prev or {}).get("baseURL", "")).strip() or str(cat.get("baseURL", "")).strip()),
                "keyEnv": (str((prev or {}).get("keyEnv", "")).strip() or str(cat.get("keyEnv", "")).strip()),
                "models": models,
                "selectedModels": selected_models,
            }
        )

    # Step 3: choose default provider/model.
    combo_choices: list[tuple[str, str]] = []
    for p in new_providers:
        label = str(p.get("label", "")).strip() or str(p.get("id", "")).strip()
        pid = str(p.get("id", "")).strip()
        for m in p.get("selectedModels", []) if isinstance(p.get("selectedModels"), list) else []:
            m = str(m).strip()
            if not m:
                continue
            combo_choices.append((f"{pid}::{m}", f"{label} / {m}"))

    prev_default = existing_cfg.get("default") if isinstance(existing_cfg.get("default"), dict) else {}
    prev_pid = str(prev_default.get("providerId", "")).strip()
    prev_model = str(prev_default.get("model", "")).strip()
    current_combo = f"{prev_pid}::{prev_model}" if prev_pid and prev_model else combo_choices[0][0]
    picked = prompt_menu("Default LLM", combo_choices, current_combo)
    if picked is None:
        return
    if "::" in picked:
        default_pid, default_model = picked.split("::", 1)
    else:
        default_pid, default_model = new_providers[0]["id"], str(new_providers[0].get("selectedModels", [""])[0])

    cfg = {"schemaVersion": 1, "providers": new_providers, "default": {"providerId": default_pid, "model": default_model}}
    save_llm_config(cfg)
    print(c(f"meta> LLM default set to {llm_default_display()}", "1;32"))

    # Optional: keys.
    update_keys = prompt_menu("Update API keys now?", [("yes", "yes"), ("no", "no")], "no")
    if update_keys == "yes":
        llm_collect_keys(cfg, [p["id"] for p in new_providers if p.get("id")])


def llm_use_default() -> None:
    if not LLM_CONFIG_FILE.exists():
        print(c("meta> llm.config.json not found; run `/llm setup` first", "33"))
        return
    cfg = load_llm_config()
    providers = cfg.get("providers", []) if isinstance(cfg.get("providers"), list) else []
    combo_choices: list[tuple[str, str]] = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", "")).strip()
        label = str(p.get("label", "")).strip() or pid
        selected = p.get("selectedModels") if isinstance(p.get("selectedModels"), list) else []
        for m in [str(x).strip() for x in selected if str(x).strip()]:
            combo_choices.append((f"{pid}::{m}", f"{label} / {m}"))
    if not combo_choices:
        print(c("meta> no models selected; run `/llm setup`", "33"))
        return
    d = cfg.get("default") if isinstance(cfg.get("default"), dict) else {}
    cur = f"{str(d.get('providerId','')).strip()}::{str(d.get('model','')).strip()}"
    picked = prompt_menu("Default LLM", combo_choices, cur)
    if picked is None:
        return
    pid, model = picked.split("::", 1)
    cfg["default"] = {"providerId": pid, "model": model}
    save_llm_config(cfg)
    print(c(f"meta> LLM default set to {llm_default_display()}", "1;32"))


def parse_agent_yaml_scalar(value: str) -> str:
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    return value


def find_vault_root_from_agent() -> Path:
    text = AGENT_FILE.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    last_source: str | None = None

    for line in lines:
        src_match = re.match(r"\s*-\s*source:\s*(.+)$", line)
        if src_match:
            last_source = parse_agent_yaml_scalar(src_match.group(1))
            continue
        tgt_match = re.match(r"\s*target:\s*(.+)$", line)
        if tgt_match and last_source is not None:
            target = parse_agent_yaml_scalar(tgt_match.group(1))
            if target == "/vault":
                return Path(last_source).expanduser().resolve()
            last_source = None

    # Safe fallback for this project layout.
    return PROJECT_DIR.parent.resolve()


def read_vault_readonly_from_agent() -> bool | None:
    """Best-effort parse of agent.claw to find readOnly for the /vault mount."""
    if not AGENT_FILE.exists():
        return None
    try:
        text = AGENT_FILE.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    in_vault_mount = False
    for line in text.splitlines():
        src_match = re.match(r"\s*-\s*source:\s*(.+)$", line)
        if src_match:
            in_vault_mount = False
            continue

        tgt_match = re.match(r"\s*target:\s*(.+)$", line)
        if tgt_match:
            target = parse_agent_yaml_scalar(tgt_match.group(1))
            in_vault_mount = target == "/vault"
            continue

        ro_match = re.match(r"\s*readOnly:\s*(.+)$", line)
        if ro_match and in_vault_mount:
            raw = parse_agent_yaml_scalar(ro_match.group(1)).strip().lower()
            if raw in {"true", "yes", "1"}:
                return True
            if raw in {"false", "no", "0"}:
                return False
            return None

    return None


def set_mount_readonly_by_target(agent_text: str, target: str, read_only: bool) -> str:
    """Set or insert `readOnly` for the mount item whose `target` matches."""
    lines = agent_text.splitlines()
    in_mounts = False
    mounts_indent = 0

    def indent_len(s: str) -> int:
        return len(s) - len(s.lstrip(" \t"))

    for i, line in enumerate(lines):
        trimmed = line.strip()
        indent = indent_len(line)

        if trimmed == "mounts:":
            in_mounts = True
            mounts_indent = indent
            continue

        if in_mounts and indent <= mounts_indent and trimmed and not trimmed.startswith("-"):
            in_mounts = False

        if not in_mounts:
            continue

        m = re.match(r"\s*target:\s*(.+)$", line)
        if not m:
            continue
        tgt = parse_agent_yaml_scalar(m.group(1))
        if tgt != target:
            continue

        desired = " " * indent + f"readOnly: {'true' if read_only else 'false'}"

        # Look ahead within this mount item for existing readOnly.
        insert_at = i + 1
        for j in range(i + 1, len(lines)):
            jline = lines[j]
            jtrim = jline.strip()
            jindent = indent_len(jline)

            # Next mount item (same list level) or leaving mounts block.
            if jindent <= mounts_indent + 2 and jtrim.startswith("-"):
                break
            if jindent <= mounts_indent and jtrim and not jtrim.startswith("-"):
                break

            if jtrim.startswith("readOnly:"):
                lines[j] = desired
                return "\n".join(lines)

        lines.insert(insert_at, desired)
        return "\n".join(lines)

    return agent_text


def sanitize_vault_relative_path(raw: str, require_md: bool = True) -> PurePosixPath:
    candidate = raw.strip().replace("\\", "/")
    if candidate == "":
        raise ValueError("path is empty")
    if candidate.startswith("/"):
        raise ValueError("path must be relative to vault")
    if candidate.startswith("~"):
        raise ValueError("home-expansion paths are not allowed")

    pure = PurePosixPath(candidate)
    if not pure.parts:
        raise ValueError("invalid path")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("path traversal is not allowed")

    root = pure.parts[0]
    if root not in ALLOWED_SAVE_ROOTS:
        raise ValueError(f"path must start with one of {', '.join(ALLOWED_SAVE_ROOTS)}")

    if require_md and pure.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError("target file must end with .md or .markdown")

    return pure


def resolve_safe_vault_path(vault_root: Path, rel_path: PurePosixPath) -> Path:
    target = (vault_root / Path(*rel_path.parts)).resolve()
    root = vault_root.resolve()
    try:
        common = os.path.commonpath([str(root), str(target)])
    except ValueError as exc:
        raise ValueError("path resolution failed") from exc
    if common != str(root):
        raise ValueError("resolved path escapes vault")
    return target


def append_write_audit(path: PurePosixPath, mode: str, overwritten: bool, content: str) -> None:
    append_audit(
        {
            "action": "save",
            "path": path.as_posix(),
            "confirmMode": mode,
            "overwritten": overwritten,
            "bytes": len(content.encode("utf-8")),
        }
    )


def append_audit(event: dict) -> None:
    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    payload.update(event)
    with WRITE_AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def confirm_write(target: Path, new_text: str, mode: str) -> bool:
    if mode == "allowlist_auto":
        return True

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(c("meta> non-interactive terminal; write confirmation required", "31"))
        return False

    if mode == "enter_once":
        prompt = f"meta> write to {target}? [Enter=yes, n=no]: "
        ans = input(c(prompt, "1;33")).strip().lower()
        return ans in {"", "y", "yes"}

    # diff_yes mode
    old_text = ""
    if target.exists():
        old_text = target.read_text(encoding="utf-8", errors="ignore")

    print(c("meta> preview diff", "1;33"))
    old_lines = old_text.splitlines(keepends=True)
    new_lines = (new_text + "\n").splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="current",
            tofile="new",
            lineterm="",
            n=2,
        )
    )
    if diff:
        for line in diff[:300]:
            if line.startswith("+") and not line.startswith("+++"):
                print(c(line, "32"))
            elif line.startswith("-") and not line.startswith("---"):
                print(c(line, "31"))
            else:
                print(line)
        if len(diff) > 300:
            print(c("... diff truncated ...", "2"))
    else:
        print(c("(no content changes)", "2"))

    ans = input(c("meta> type 'yes' to confirm write: ", "1;33")).strip().lower()
    return ans == "yes"


def build_default_save_path(default_dir: str) -> PurePosixPath:
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return PurePosixPath(default_dir) / f"note-{stamp}.md"


def parse_save_args(args: list[str], state: SessionState, defaults: dict[str, str]) -> tuple[PurePosixPath | None, bool]:
    persist = False
    target: str | None = None
    pending_default_dir = None

    i = 0
    while i < len(args):
        token = args[i]
        if token == "--default":
            persist = True
            i += 1
            continue
        if token == "--default-dir":
            if i + 1 >= len(args):
                raise ValueError("--default-dir requires a value")
            pending_default_dir = args[i + 1]
            i += 2
            continue
        if token == "--to":
            if i + 1 >= len(args):
                raise ValueError("--to requires a value")
            target = args[i + 1]
            i += 2
            continue
        if token == "--help":
            raise ValueError("help")
        if token.startswith("--"):
            raise ValueError(f"unknown option: {token}")
        if target is None:
            target = token
        else:
            raise ValueError("multiple target paths provided")
        i += 1

    if pending_default_dir is not None:
        normalized = sanitize_default_dir(pending_default_dir)
        state.save_default_dir = normalized
        print(c(f"meta> save_default_dir set to {normalized}", "1;32"))
        if persist:
            defaults["save_default_dir"] = normalized
            save_defaults(defaults)
            print(c("meta> saved as project default", "2"))

    if target is None:
        if pending_default_dir is not None:
            return None, persist
        return build_default_save_path(state.save_default_dir), persist

    return sanitize_vault_relative_path(target, require_md=True), persist


def do_save_last_response(args: list[str], state: SessionState, defaults: dict[str, str]) -> None:
    try:
        rel_path, _ = parse_save_args(args, state, defaults)
    except ValueError as exc:
        msg = str(exc)
        if msg == "help":
            print(c("meta> usage: /save <Research/.../file.md> [--default]", "2"))
            print(c("meta>        /save --to <Learning/.../file.md>", "2"))
            print(c("meta>        /save --default-dir <Research/...> [--default]", "2"))
            return
        print(c(f"meta> {msg}", "31"))
        return

    if rel_path is None:
        return

    response = read_response_text()
    if response.startswith("[bot error]"):
        print(c("meta> cannot save: last response is an error", "31"))
        return
    if response.strip() == "":
        print(c("meta> cannot save: last response is empty", "31"))
        return

    vault_root = find_vault_root_from_agent()
    try:
        target_path = resolve_safe_vault_path(vault_root, rel_path)
    except ValueError as exc:
        print(c(f"meta> {exc}", "31"))
        return

    mode = normalize_confirm_mode(state.write_confirm_mode)
    overwritten = target_path.exists()
    if not confirm_write(target_path, response, mode):
        print(c("meta> save cancelled", "33"))
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = response if response.endswith("\n") else response + "\n"
    target_path.write_text(content, encoding="utf-8")
    append_write_audit(rel_path, mode, overwritten, content)
    print(c(f"meta> saved {rel_path.as_posix()}", "1;32"))


def do_append_last_response(args: list[str], state: SessionState) -> None:
    if not args:
        print(c("meta> usage: /append <Research/.../file.md>", "2"))
        return

    try:
        rel_path = sanitize_vault_relative_path(args[0], require_md=True)
    except ValueError as exc:
        print(c(f"meta> {exc}", "31"))
        return

    response = read_response_text()
    if response.startswith("[bot error]"):
        print(c("meta> cannot append: last response is an error", "31"))
        return
    if response.strip() == "":
        print(c("meta> cannot append: last response is empty", "31"))
        return

    vault_root = find_vault_root_from_agent()
    try:
        target_path = resolve_safe_vault_path(vault_root, rel_path)
    except ValueError as exc:
        print(c(f"meta> {exc}", "31"))
        return

    old_text = ""
    if target_path.exists():
        if target_path.is_dir():
            print(c("meta> target exists but is a directory", "31"))
            return
        old_text = target_path.read_text(encoding="utf-8", errors="ignore")

    # Append with a separator for readability.
    sep = "\n\n" if old_text.strip() else ""
    new_text = old_text.rstrip("\n") + sep + response.strip() + "\n"

    mode = normalize_confirm_mode(state.write_confirm_mode)
    overwritten = target_path.exists()
    if not confirm_write(target_path, new_text, mode):
        print(c("meta> append cancelled", "33"))
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(new_text, encoding="utf-8")
    append_audit(
        {
            "action": "append",
            "path": rel_path.as_posix(),
            "confirmMode": mode,
            "overwritten": overwritten,
            "bytes": len(new_text.encode("utf-8")),
        }
    )
    print(c(f"meta> appended to {rel_path.as_posix()}", "1;32"))


def do_touch(args: list[str], state: SessionState) -> None:
    if not args:
        print(c("meta> usage: /touch <Research/.../file.md>", "2"))
        return

    try:
        rel_path = sanitize_vault_relative_path(args[0], require_md=True)
    except ValueError as exc:
        print(c(f"meta> {exc}", "31"))
        return

    vault_root = find_vault_root_from_agent()
    try:
        target_path = resolve_safe_vault_path(vault_root, rel_path)
    except ValueError as exc:
        print(c(f"meta> {exc}", "31"))
        return

    if target_path.exists():
        if target_path.is_dir():
            print(c("meta> target exists but is a directory", "31"))
            return
        print(c(f"meta> already exists: {rel_path.as_posix()}", "2"))
        return

    mode = normalize_confirm_mode(state.write_confirm_mode)
    if not confirm_write(target_path, "", mode):
        print(c("meta> touch cancelled", "33"))
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("", encoding="utf-8")
    append_audit(
        {
            "action": "touch",
            "path": rel_path.as_posix(),
        }
    )
    print(c(f"meta> created {rel_path.as_posix()}", "1;32"))


def main() -> int:
    ensure_paths()
    setup_readline()
    metaclaw_bin = resolve_metaclaw_bin()

    defaults = load_defaults()
    state = SessionState(
        network_mode=normalize_network_mode(defaults["network_mode"]),
        render_mode=normalize_render_mode(defaults["render_mode"]),
        vault_access=normalize_vault_access(defaults["vault_access"]),
        retrieval_scope=normalize_scope(defaults["retrieval_scope"]),
        write_confirm_mode=normalize_confirm_mode(defaults["write_confirm_mode"]),
        save_default_dir=sanitize_default_dir(defaults["save_default_dir"]),
        output_focus=normalize_focus(defaults.get("output_focus", DEFAULT_VALUES["output_focus"])),
        tool_max_steps=normalize_tool_max_steps(defaults.get("tool_max_steps", DEFAULT_VALUES["tool_max_steps"])),
        tool_max_calls_per_step=normalize_tool_max_calls_per_step(
            defaults.get("tool_max_calls_per_step", DEFAULT_VALUES["tool_max_calls_per_step"])
        ),
    )

    # Environment overrides are session-only.
    state.network_mode = normalize_network_mode(os.getenv("BOT_NETWORK_MODE", state.network_mode))
    state.render_mode = normalize_render_mode(os.getenv("BOT_RENDER_MODE", state.render_mode))
    state.vault_access = normalize_vault_access(os.getenv("BOT_VAULT_ACCESS", state.vault_access))
    state.output_focus = normalize_focus(os.getenv("BOT_OUTPUT_FOCUS", state.output_focus))
    state.tool_max_steps = normalize_tool_max_steps(os.getenv("BOT_TOOL_MAX_STEPS", state.tool_max_steps))
    state.tool_max_calls_per_step = normalize_tool_max_calls_per_step(
        os.getenv("BOT_TOOL_MAX_CALLS_PER_STEP", state.tool_max_calls_per_step)
    )

    print_banner(metaclaw_bin, state)

    while True:
        try:
            prompt = input(c("you> ", "1;36"))
        except EOFError:
            print("")
            break
        except KeyboardInterrupt:
            print("")
            continue

        text = prompt.strip()
        if not text:
            continue

        if text in {"/exit", "/quit"}:
            break
        if text == "/help":
            print_help()
            continue
        if text == "/clear":
            clear_screen()
            print_banner(metaclaw_bin, state)
            continue
        if text == "/status":
            local_status(state)
            continue
        if text == "/refresh":
            refresh_context()
            continue
        if text == "/history":
            show_local_history()
            continue
        if text == "/reset":
            reset_memory()
            continue

        if text.startswith("/"):
            tokens = parse_command_tokens(text)
            if not tokens:
                continue
            cmd = tokens[0].lower()
            args = tokens[1:]

            if cmd == "/new":
                name = " ".join(args).strip() if args else ""
                sid = sanitize_session_id(name)
                if not sid:
                    sid = dt.datetime.now(dt.timezone.utc).strftime("s-%Y%m%dT%H%M%SZ")
                activate_runtime_session(sid, migrate_existing=False)
                refresh_context()
                print(c(f"meta> active session: {sid}", "2"))
                print_banner(metaclaw_bin, state)
                continue

            if cmd == "/refresh":
                refresh_context()
                continue

            if cmd in {"/llm", "/model"}:
                sub = args[0].lower() if args else ""
                if sub in {"", "status"}:
                    llm_show_status()
                    continue
                if sub == "setup":
                    llm_setup_interactive()
                    continue
                if sub == "use":
                    llm_use_default()
                    continue
                if sub == "key":
                    if not LLM_CONFIG_FILE.exists():
                        print(c("meta> llm.config.json not found; run `/llm setup` first", "33"))
                        continue
                    cfg = load_llm_config()
                    provider_ids = [str(p.get("id", "")).strip() for p in cfg.get("providers", []) if isinstance(p, dict) and str(p.get("id", "")).strip()]
                    llm_collect_keys(cfg, provider_ids)
                    continue
                print(c("meta> usage: /llm [status|setup|use|key]", "31"))
                continue

            if cmd == "/render":
                if args and args[0].lower() == "demo":
                    used_mode = render_mode_demo(state.render_mode)
                    if used_mode != state.render_mode:
                        print(c(f"\nmeta> glow render failed, used {used_mode}", "33"))
                    continue

                value, persist = parse_value_and_default(args)
                selected = None
                if value is None:
                    selected = prompt_select("render", RENDER_CHOICES, state.render_mode)
                else:
                    selected = value.lower()
                if selected is None:
                    continue
                if selected not in {"plain", "glow"}:
                    print(c("meta> usage: /render plain|glow|demo [--default]", "31"))
                    continue
                if selected == "glow" and not has_glow():
                    print(c("meta> glow is not installed, staying in plain mode", "33"))
                    selected = "plain"
                apply_mode_change(state, defaults, "render_mode", selected, persist)
                continue

            if cmd == "/focus":
                value, persist = parse_value_and_default(args)
                selected = None
                if value is None:
                    selected = prompt_select("focus", FOCUS_CHOICES, state.output_focus)
                else:
                    selected = value.lower()
                if selected is None:
                    continue
                selected = normalize_focus(selected)
                if selected not in {"start", "end", "stay", "input"}:
                    print(c("meta> usage: /focus start|end|stay|input [--default]", "31"))
                    continue
                apply_mode_change(state, defaults, "output_focus", selected, persist)
                continue

            if cmd == "/show":
                where = "start"
                if args:
                    where = args[0].strip().lower()
                if where not in {"start", "end"}:
                    print(c("meta> usage: /show [start|end]", "31"))
                    continue
                show_last_response(state, start_at_end=(where == "end"))
                continue

            if cmd in {"/net", "/network"}:
                value, persist = parse_value_and_default(args)
                selected = None
                if value is None:
                    selected = prompt_select("network", NETWORK_CHOICES, state.network_mode)
                else:
                    selected = value.lower()
                if selected is None:
                    continue
                selected = normalize_network_mode(selected)
                if selected not in {"none", "outbound"}:
                    print(c("meta> usage: /net none|out [--default]", "31"))
                    continue
                apply_mode_change(state, defaults, "network_mode", selected, persist)
                continue

            if cmd == "/vault":
                value, persist = parse_value_and_default(args)
                selected = None
                if value is None:
                    selected = prompt_select("vault", VAULT_CHOICES, state.vault_access)
                else:
                    selected = value.lower()
                if selected is None:
                    continue
                selected = normalize_vault_access(selected)
                if selected not in {"ro", "rw"}:
                    print(c("meta> usage: /vault ro|rw [--default]", "31"))
                    continue
                if selected == "rw":
                    print(c("meta> WARNING: vault will be mounted read-write inside the container (less safe)", "33"))
                apply_mode_change(state, defaults, "vault_access", selected, persist)
                continue

            if cmd == "/scope":
                value, persist = parse_value_and_default(args)
                selected = None
                if value is None:
                    selected = prompt_select("scope", SCOPE_CHOICES, state.retrieval_scope)
                else:
                    selected = value.lower()
                if selected is None:
                    continue
                if selected not in {"limited", "all"}:
                    print(c("meta> usage: /scope limited|all [--default]", "31"))
                    continue
                apply_mode_change(state, defaults, "retrieval_scope", selected, persist)
                continue

            if cmd == "/confirm":
                value, persist = parse_value_and_default(args)
                selected = None
                if value is None:
                    selected = prompt_select("confirm", CONFIRM_CHOICES, state.write_confirm_mode)
                else:
                    selected = value.lower()
                if selected is None:
                    continue
                selected = normalize_confirm_mode(selected)
                if selected not in {"enter_once", "diff_yes", "allowlist_auto"}:
                    print(c("meta> usage: /confirm once|diff|auto [--default]", "31"))
                    continue
                apply_mode_change(state, defaults, "write_confirm_mode", selected, persist)
                continue

            if cmd == "/steps":
                value, persist = parse_value_and_default(args)
                if value is None:
                    n = prompt_int("Tool max steps", current=state.tool_max_steps, min_value=1, max_value=128)
                else:
                    try:
                        n = int(value)
                    except Exception:
                        n = None
                if n is None:
                    print(c("meta> usage: /steps <n> [--default]", "31"))
                    continue
                state.tool_max_steps = normalize_tool_max_steps(n)
                print(c(f"meta> tool_max_steps set to {state.tool_max_steps}", "1;32"))
                if persist:
                    defaults["tool_max_steps"] = state.tool_max_steps
                    save_defaults(defaults)
                    print(c("meta> saved as project default", "2"))
                continue

            if cmd == "/calls":
                value, persist = parse_value_and_default(args)
                if value is None:
                    n = prompt_int(
                        "Tool max calls per step",
                        current=state.tool_max_calls_per_step,
                        min_value=1,
                        max_value=24,
                    )
                else:
                    try:
                        n = int(value)
                    except Exception:
                        n = None
                if n is None:
                    print(c("meta> usage: /calls <n> [--default]", "31"))
                    continue
                state.tool_max_calls_per_step = normalize_tool_max_calls_per_step(n)
                print(c(f"meta> tool_max_calls_per_step set to {state.tool_max_calls_per_step}", "1;32"))
                if persist:
                    defaults["tool_max_calls_per_step"] = state.tool_max_calls_per_step
                    save_defaults(defaults)
                    print(c("meta> saved as project default", "2"))
                continue

            if cmd == "/web":
                query = " ".join(args).strip()
                if not query:
                    print(c("meta> usage: /web <query>", "31"))
                    continue
                web_env = get_web_key_env_name()
                if not os.getenv(web_env):
                    print(c(f"meta> missing web search key env: {web_env}", "33"))
                    print(c(f"meta> set it to enable Tavily: export {web_env}=your_key", "2"))
                    continue

                # Web lookup requires outbound, but keep your session default unchanged.
                prev_mode = state.network_mode
                state.network_mode = "outbound"
                try:
                    web_prompt = "/web " + query
                    if not check_key_env_or_warn(web_prompt):
                        continue
                    ok, elapsed = metaclaw_run(metaclaw_bin, web_prompt, state)
                    if not ok:
                        print(c("bot> run failed", "31"))
                        for line in tail_log(RUN_LOG_FILE, n=40):
                            print(f"  {line}")
                        response_text = read_response_text()
                        if response_text.startswith("[bot error]"):
                            print(f"  {response_text}")
                        continue
                    response_text = read_response_text()
                    used_mode = present_bot_response(response_text, elapsed, state)
                    if state.render_mode == "glow" and looks_like_markdown(response_text) and used_mode != state.render_mode:
                        print(c(f"\nmeta> glow render failed, used {used_mode}", "33"))
                finally:
                    state.network_mode = prev_mode
                continue

            if cmd == "/save":
                do_save_last_response(args, state, defaults)
                continue
            if cmd == "/append":
                do_append_last_response(args, state)
                continue
            if cmd == "/touch":
                do_touch(args, state)
                continue

        if state.network_mode == "none" and prompt_uses_llm(text):
            print(c("meta> network mode is none; this query needs outbound access", "33"))
            print(c("meta> run /net out to enable LLM and web lookup", "2"))
            continue

        if not check_key_env_or_warn(text):
            continue

        ok, elapsed = metaclaw_run(metaclaw_bin, text, state)
        if not ok:
            print(c("bot> run failed", "31"))
            for line in tail_log(RUN_LOG_FILE, n=40):
                print(f"  {line}")
            response_text = read_response_text()
            if response_text.startswith("[bot error]"):
                print(f"  {response_text}")
            continue

        response_text = read_response_text()
        used_mode = present_bot_response(response_text, elapsed, state)
        if state.render_mode == "glow" and looks_like_markdown(response_text) and used_mode != state.render_mode:
            print(c(f"\nmeta> glow render failed, used {used_mode}", "33"))

    save_readline_history()
    print(c("bye", "2"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
