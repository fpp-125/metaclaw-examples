#!/usr/bin/env python3
"""Improved terminal UI for MetaClaw Obsidian bot."""

from __future__ import annotations

import datetime as dt
import difflib
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
WRITE_AUDIT_FILE = HOST_LOG_DIR / "write_audit.jsonl"

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


@dataclass
class SessionState:
    network_mode: str
    render_mode: str
    vault_access: str
    retrieval_scope: str
    write_confirm_mode: str
    save_default_dir: str


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
    REQUEST_FILE.touch(exist_ok=True)

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
    }
    DEFAULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULTS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_key_env_name() -> str:
    return os.getenv("LLM_KEY_ENV", "GEMINI_API_KEY")


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


def key_is_ready() -> bool:
    return bool(os.getenv(get_key_env_name()))


def web_key_is_ready() -> bool:
    return bool(os.getenv(get_web_key_env_name()))


def print_banner(metaclaw_bin: str, state: SessionState) -> None:
    rows = [
        f"{c('Project:', '1;33')} {PROJECT_DIR}",
        f"{c('Agent:', '1;33')}   {AGENT_FILE.name}",
        f"{c('Runtime:', '1;33')} {get_runtime_target()}",
        f"{c('Network:', '1;33')} {network_mode_label(state.network_mode)}",
        f"{c('Vault:', '1;33')}   {vault_access_label(state.vault_access)}",
        f"{c('Scope:', '1;33')}   {state.retrieval_scope}",
        f"{c('Confirm:', '1;33')} {confirm_mode_label(state.write_confirm_mode)}",
        f"{c('Render:', '1;33')}  {state.render_mode}" + (" (glow ready)" if has_glow() else ""),
        f"{c('LLM key:', '1;33')} {get_key_env_name()} {'(set)' if key_is_ready() else '(missing)'}",
        f"{c('Web key:', '1;33')} {get_web_key_env_name()} {'(set)' if web_key_is_ready() else '(missing)'}",
        f"{c('Host data:', '1;33')} {HOST_DATA_DIR}",
        f"{c('MetaClaw:', '1;33')} {metaclaw_bin}",
        "",
        f"{c('Commands:', '1;32')} /help /status /model /history /render /net /vault /scope /confirm /save /append /touch /reset /clear /exit",
    ]
    boxed("MetaClaw Obsidian Terminal Bot", rows)


def print_help() -> None:
    rows = [
        "Type any prompt and press Enter.",
        "",
        f"{c('/help', '1;32')}       Show this help",
        f"{c('/status', '1;32')}     Show runtime/key/model status",
        f"{c('/model', '1;32')}      Ask agent to report configured model",
        f"{c('/history', '1;32')}    Show recent local conversation turns",
        f"{c('/render', '1;32')}     Show or set render mode (/render plain|glow|demo [--default])",
        f"{c('/net', '1;32')}        Show or set network mode (/net none|out [--default])",
        f"{c('/vault', '1;32')}      Show or set vault access (/vault ro|rw [--default])",
        f"{c('/scope', '1;32')}      Show or set retrieval scope (/scope limited|all [--default])",
        f"{c('/confirm', '1;32')}    Show or set write confirm mode (/confirm once|diff|auto [--default])",
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
        "/web <query>  force Tavily web lookup + summarize",
        "",
        "Runtime override example:",
        "RUNTIME_TARGET=auto ./chat.sh",
        "Config defaults example:",
        "/scope limited --default",
        "/confirm once --default",
        "Key env override example:",
        "LLM_KEY_ENV=OPENAI_API_KEY ./chat.sh",
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


def write_effective_agent(network_mode: str, vault_access: str) -> Path:
    raw = AGENT_FILE.read_text(encoding="utf-8")
    if network_mode == "none":
        updated = raw.replace("mode: outbound", "mode: none", 1)
    else:
        updated = raw.replace("mode: none", "mode: outbound", 1)

    if updated == raw and f"mode: {network_mode}" not in raw:
        raise RuntimeError("cannot resolve habitat.network.mode in agent.claw")

    updated = set_mount_readonly_by_target(updated, "/vault", normalize_vault_access(vault_access) == "ro")

    EFFECTIVE_AGENT_FILE.write_text(updated, encoding="utf-8")
    return EFFECTIVE_AGENT_FILE


def write_session_config(state: SessionState) -> None:
    payload = {
        "retrieval_scope": state.retrieval_scope,
    }
    SESSION_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metaclaw_run(metaclaw_bin: str, prompt: str, state: SessionState) -> tuple[bool, float]:
    REQUEST_FILE.write_text(prompt, encoding="utf-8")
    write_session_config(state)
    if RESPONSE_FILE.exists():
        RESPONSE_FILE.unlink()

    agent_file = write_effective_agent(state.network_mode, state.vault_access)

    run_args = [
        metaclaw_bin,
        "run",
        str(agent_file),
        f"--state-dir={STATE_DIR}",
        f"--llm-api-key-env={get_key_env_name()}",
    ]
    web_key_env = get_web_key_env_name()
    if os.getenv(web_key_env):
        run_args.append(f"--secret-env={web_key_env}")

    runtime_target = get_runtime_target()
    if runtime_target != "auto":
        run_args.append(f"--runtime={runtime_target}")

    with RUN_LOG_FILE.open("w", encoding="utf-8") as lf:
        start = time.time()
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
        elapsed = time.time() - start
    return rc == 0, elapsed


def local_status(state: SessionState) -> None:
    rows = [
        f"{c('Runtime target:', '1;33')} {get_runtime_target()}",
        f"{c('Network mode:', '1;33')} {network_mode_label(state.network_mode)}",
        f"{c('Vault access:', '1;33')} {vault_access_label(state.vault_access)}",
        f"{c('Scope mode:', '1;33')} {state.retrieval_scope}",
        f"{c('Confirm mode:', '1;33')} {confirm_mode_label(state.write_confirm_mode)}",
        f"{c('Render mode:', '1;33')} {state.render_mode}" + (" (glow ready)" if has_glow() else " (glow missing)"),
        f"{c('Save default dir:', '1;33')} {state.save_default_dir}",
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
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
    if RESPONSE_FILE.exists():
        RESPONSE_FILE.unlink()
    print(c("meta> memory reset", "1;32"))


def read_response_text() -> str:
    if not RESPONSE_FILE.exists():
        return "[bot error] response file not generated"
    text = RESPONSE_FILE.read_text(encoding="utf-8", errors="ignore").rstrip("\n")
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text


def prompt_uses_llm(prompt: str) -> bool:
    return prompt.strip() != "/model"


def check_key_env_or_warn(prompt: str) -> bool:
    key_env = get_key_env_name()
    if not os.getenv(key_env):
        print(c(f"meta> missing LLM API key env: {key_env}", "31"))
        print(c(f"meta> example: export {key_env}=your_key", "2"))
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
    try:
        return shlex.split(text)
    except Exception as exc:
        print(c(f"meta> command parse error: {exc}", "31"))
        return []


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
        save_defaults(defaults)
        print(c("meta> saved as project default", "2"))


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
    )

    # Environment overrides are session-only.
    state.network_mode = normalize_network_mode(os.getenv("BOT_NETWORK_MODE", state.network_mode))
    state.render_mode = normalize_render_mode(os.getenv("BOT_RENDER_MODE", state.render_mode))
    state.vault_access = normalize_vault_access(os.getenv("BOT_VAULT_ACCESS", state.vault_access))

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
        used_mode = pretty_print_bot_response(response_text, elapsed, state.render_mode)
        if state.render_mode == "glow" and looks_like_markdown(response_text) and used_mode != state.render_mode:
            print(c(f"\nmeta> glow render failed, used {used_mode}", "33"))

    save_readline_history()
    print(c("bye", "2"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
