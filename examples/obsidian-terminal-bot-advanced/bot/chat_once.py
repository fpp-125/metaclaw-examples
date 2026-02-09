#!/usr/bin/env python3
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import random
import urllib.error
import urllib.request
from pathlib import Path

RUNTIME_DIR = Path(os.getenv("METACLAW_RUNTIME_DIR", "/runtime"))
VAULT_DIR = Path(os.getenv("OBSIDIAN_VAULT_DIR", "/vault"))
LOG_DIR = Path(os.getenv("METACLAW_LOG_DIR", "/logs"))
WORKSPACE_DIR = Path(os.getenv("METACLAW_WORKSPACE_DIR", "/workspace"))
REQUEST_FILE = RUNTIME_DIR / "request.txt"
RESPONSE_FILE = RUNTIME_DIR / "response.txt"
HISTORY_FILE = RUNTIME_DIR / "history.json"
SESSION_FILE = RUNTIME_DIR / "session.json"
AGENTS_CONTEXT_FILE = RUNTIME_DIR / "agents_context.md"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_API_URL = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search").strip()
LIMITED_SCOPE_PREFIXES = ("Research/", "Learning/")
TOOL_AUDIT_FILE = LOG_DIR / "tool_audit.jsonl"

STOPWORDS = {
    "the", "this", "that", "with", "from", "have", "what", "when", "where", "which", "about",
    "into", "your", "you", "and", "for", "are", "how", "can", "use", "using", "should", "please",
    "在", "这个", "那个", "怎么", "可以", "一下", "关于", "以及", "需要", "帮我", "一个",
}

TOOL_MAX_STEPS = int(os.getenv("BOT_TOOL_MAX_STEPS", "6").strip() or "6")
TOOL_TIMEOUT_SEC = float(os.getenv("BOT_TOOL_TIMEOUT_SEC", "5").strip() or "5")
TOOL_MAX_OUTPUT_BYTES = int(os.getenv("BOT_TOOL_MAX_OUTPUT_BYTES", "32768").strip() or "32768")
TOOL_MAX_READ_BYTES = int(os.getenv("BOT_TOOL_MAX_READ_BYTES", "262144").strip() or "262144")
TOOL_MAX_WRITE_BYTES = int(os.getenv("BOT_TOOL_MAX_WRITE_BYTES", "524288").strip() or "524288")
TOOL_MAX_SEARCH_FILES = int(os.getenv("BOT_TOOL_MAX_SEARCH_FILES", "800").strip() or "800")

LLM_MAX_RETRIES = int(os.getenv("BOT_LLM_MAX_RETRIES", "4").strip() or "4")
LLM_RETRY_BASE_SEC = float(os.getenv("BOT_LLM_RETRY_BASE_SEC", "0.8").strip() or "0.8")
LLM_RETRY_MAX_SEC = float(os.getenv("BOT_LLM_RETRY_MAX_SEC", "6.0").strip() or "6.0")

AGENT_SPAWN_MAX = int(os.getenv("BOT_AGENT_SPAWN_MAX", "3").strip() or "3")

DEFAULT_ALLOW_CMDS = "ls,find,grep"
TOOL_ALLOW_CMDS = {
    c.strip()
    for c in (os.getenv("BOT_TOOL_ALLOW_CMDS", DEFAULT_ALLOW_CMDS) or DEFAULT_ALLOW_CMDS).split(",")
    if c.strip()
}

_agent_spawn_count = 0


def read_agents_context() -> str:
    # Best-effort: a missing file should not fail the bot.
    try:
        if AGENTS_CONTEXT_FILE.exists():
            return AGENTS_CONTEXT_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        pass
    return ""

DEFAULT_ALLOWED_ROOTS = "/vault,/workspace"
TOOL_ALLOWED_ROOTS = [
    Path(p.strip())
    for p in (os.getenv("BOT_TOOL_ALLOWED_ROOTS", DEFAULT_ALLOWED_ROOTS) or DEFAULT_ALLOWED_ROOTS).split(",")
    if p.strip()
]

DEFAULT_WRITE_ROOTS = "/vault,/workspace"
TOOL_WRITE_ROOTS = [
    Path(p.strip())
    for p in (os.getenv("BOT_TOOL_WRITE_ROOTS", DEFAULT_WRITE_ROOTS) or DEFAULT_WRITE_ROOTS).split(",")
    if p.strip()
]


def audit_tool(event: dict) -> None:
    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    payload.update(event)
    try:
        TOOL_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TOOL_AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort: audit should never fail the run.
        pass


def truncate_bytes(text: str, max_bytes: int) -> tuple[str, bool]:
    b = text.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return text, False
    cut = b[:max_bytes]
    # Avoid splitting multi-byte sequences.
    out = cut.decode("utf-8", errors="ignore")
    return out + "\n...[truncated]...\n", True


def normalize_path_input(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("vault/"):
        return "/" + raw
    if raw.startswith("workspace/"):
        return "/" + raw
    return raw


def resolve_under_allowed_roots(path_str: str, *, for_write: bool) -> Path:
    value = normalize_path_input(path_str)
    if value == "":
        raise ValueError("path is empty")

    # Prefer absolute paths. If relative, interpret relative to /workspace.
    if value.startswith("/"):
        candidate = Path(value)
    else:
        candidate = (WORKSPACE_DIR / value)

    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate.absolute()

    roots = TOOL_WRITE_ROOTS if for_write else TOOL_ALLOWED_ROOTS
    resolved_roots: list[Path] = []
    for r in roots:
        try:
            resolved_roots.append(r.resolve())
        except Exception:
            resolved_roots.append(r.absolute())

    for root in resolved_roots:
        try:
            common = os.path.commonpath([str(root), str(resolved)])
        except ValueError:
            continue
        if common == str(root):
            return resolved

    scope = "write" if for_write else "read"
    allowed = ", ".join(str(r) for r in roots)
    raise ValueError(f"path is outside allowed {scope} roots ({allowed})")


def tool_fs_list_dir(args: dict) -> dict:
    path = str(args.get("path", "")).strip()
    depth = int(args.get("depth", 1) or 1)
    max_entries = int(args.get("max_entries", 200) or 200)
    if depth < 1:
        depth = 1
    if depth > 4:
        depth = 4
    if max_entries < 1:
        max_entries = 1
    if max_entries > 1000:
        max_entries = 1000

    p = resolve_under_allowed_roots(path, for_write=False)
    if not p.exists():
        return {"ok": False, "error": "path does not exist"}
    if not p.is_dir():
        return {"ok": False, "error": "path is not a directory"}

    results = []
    base = p
    try:
        base_rel = base
    except Exception:
        base_rel = base

    # Walk with depth control.
    queue: list[tuple[Path, int]] = [(p, 0)]
    while queue and len(results) < max_entries:
        cur, d = queue.pop(0)
        try:
            entries = sorted(cur.iterdir(), key=lambda x: x.name.lower())
        except Exception:
            continue
        for e in entries:
            if e.name.startswith("."):
                continue
            try:
                rel = e.relative_to(base)
            except Exception:
                rel = e
            item = {
                "path": str(rel),
                "type": "dir" if e.is_dir() else "file",
            }
            results.append(item)
            if len(results) >= max_entries:
                break
            if e.is_dir() and d + 1 < depth:
                queue.append((e, d + 1))

    return {"ok": True, "entries": results, "base": str(base_rel)}


def tool_fs_read_file(args: dict) -> dict:
    path = str(args.get("path", "")).strip()
    max_bytes = int(args.get("max_bytes", TOOL_MAX_READ_BYTES) or TOOL_MAX_READ_BYTES)
    if max_bytes < 1:
        max_bytes = TOOL_MAX_READ_BYTES
    if max_bytes > 1024 * 1024:
        max_bytes = 1024 * 1024
    p = resolve_under_allowed_roots(path, for_write=False)
    if not p.exists() or not p.is_file():
        return {"ok": False, "error": "file not found"}
    try:
        raw = p.read_bytes()
    except Exception as exc:
        return {"ok": False, "error": f"read failed: {exc}"}
    clipped = raw[:max_bytes]
    text = clipped.decode("utf-8", errors="ignore")
    truncated = len(raw) > max_bytes
    if truncated:
        text += "\n...[truncated]...\n"
    return {"ok": True, "path": str(p), "text": text, "truncated": truncated, "bytes": len(raw)}


def tool_fs_write_file(args: dict) -> dict:
    path = str(args.get("path", "")).strip()
    content = args.get("content", "")
    create_dirs = bool(args.get("create_dirs", True))
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    clipped, truncated = truncate_bytes(content, TOOL_MAX_WRITE_BYTES)
    p = resolve_under_allowed_roots(path, for_write=True)
    try:
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(clipped, encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "error": f"write failed: {exc}"}
    return {"ok": True, "path": str(p), "bytes": len(clipped.encode('utf-8')), "truncated": truncated}


def tool_fs_append_file(args: dict) -> dict:
    path = str(args.get("path", "")).strip()
    content = args.get("content", "")
    create_dirs = bool(args.get("create_dirs", True))
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    clipped, truncated = truncate_bytes(content, TOOL_MAX_WRITE_BYTES)
    p = resolve_under_allowed_roots(path, for_write=True)
    try:
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(clipped)
    except Exception as exc:
        return {"ok": False, "error": f"append failed: {exc}"}
    return {"ok": True, "path": str(p), "bytes": len(clipped.encode('utf-8')), "truncated": truncated}


def tool_fs_touch(args: dict) -> dict:
    path = str(args.get("path", "")).strip()
    create_dirs = bool(args.get("create_dirs", True))
    p = resolve_under_allowed_roots(path, for_write=True)
    try:
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("", encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "error": f"touch failed: {exc}"}
    return {"ok": True, "path": str(p)}


def tool_fs_search(args: dict) -> dict:
    root = str(args.get("root", "")).strip() or "/vault"
    query = str(args.get("query", "")).strip()
    glob = str(args.get("glob", "*.md")).strip() or "*.md"
    max_files = int(args.get("max_files", TOOL_MAX_SEARCH_FILES) or TOOL_MAX_SEARCH_FILES)
    max_matches = int(args.get("max_matches", 80) or 80)
    if query == "":
        return {"ok": False, "error": "query is empty"}
    if max_files < 1:
        max_files = TOOL_MAX_SEARCH_FILES
    if max_files > 5000:
        max_files = 5000
    if max_matches < 1:
        max_matches = 1
    if max_matches > 500:
        max_matches = 500

    root_path = resolve_under_allowed_roots(root, for_write=False)
    if not root_path.exists() or not root_path.is_dir():
        return {"ok": False, "error": "root is not a directory"}

    matches = []
    scanned = 0
    q = query.lower()
    for path in root_path.rglob(glob):
        if scanned >= max_files:
            break
        scanned += 1
        try:
            if path.is_dir():
                continue
            # Skip very large files.
            if path.stat().st_size > 2_000_000:
                continue
        except Exception:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lower = text.lower()
        idx = lower.find(q)
        if idx < 0:
            continue
        start = max(0, idx - 120)
        end = min(len(text), idx + 240)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        try:
            rel = path.relative_to(root_path).as_posix()
        except Exception:
            rel = str(path)
        matches.append({"file": rel, "snippet": snippet})
        if len(matches) >= max_matches:
            break

    return {"ok": True, "root": str(root_path), "query": query, "scanned": scanned, "matches": matches}


def tool_cmd_exec(args: dict) -> dict:
    argv = args.get("argv")
    cwd = str(args.get("cwd", "")).strip() or str(WORKSPACE_DIR)
    timeout = float(args.get("timeout_sec", TOOL_TIMEOUT_SEC) or TOOL_TIMEOUT_SEC)
    if timeout < 0.2:
        timeout = 0.2
    if timeout > 30:
        timeout = 30

    if not isinstance(argv, list) or not argv:
        return {"ok": False, "error": "argv must be a non-empty list"}
    argv = [str(x) for x in argv]
    cmd = argv[0].strip()
    if cmd not in TOOL_ALLOW_CMDS:
        return {"ok": False, "error": f"command not allowed: {cmd} (allowed: {', '.join(sorted(TOOL_ALLOW_CMDS))})"}

    cwd_path = resolve_under_allowed_roots(cwd, for_write=False)

    # Basic guardrail: reject absolute path args outside allowed roots.
    for a in argv[1:]:
        a = str(a)
        if a.startswith("/"):
            try:
                _ = resolve_under_allowed_roots(a, for_write=False)
            except Exception:
                return {"ok": False, "error": f"arg path outside allowed roots: {a}"}

    # Sanitize env for spawned commands to reduce accidental secret leakage via printenv, etc.
    env = {
        "PATH": os.getenv("PATH", ""),
        "LANG": os.getenv("LANG", "C.UTF-8"),
        "LC_ALL": os.getenv("LC_ALL", "C.UTF-8"),
        "HOME": os.getenv("HOME", "/"),
        "PWD": str(cwd_path),
    }

    start = time.time()
    try:
        res = subprocess.run(
            argv,
            cwd=str(cwd_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=False,
        )
        out = res.stdout or b""
        truncated = False
        if len(out) > TOOL_MAX_OUTPUT_BYTES:
            out = out[:TOOL_MAX_OUTPUT_BYTES]
            truncated = True
        text = out.decode("utf-8", errors="ignore")
        if truncated:
            text += "\n...[truncated]...\n"
        return {
            "ok": True,
            "exit_code": res.returncode,
            "output": text,
            "truncated": truncated,
            "elapsed_sec": round(time.time() - start, 3),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    except Exception as exc:
        return {"ok": False, "error": f"exec failed: {exc}"}


def tool_agent_spawn(args: dict) -> dict:
    """Spawn a reasoning-only sub-agent (no file tools).

    This is implemented as an additional LLM call with a tighter system prompt.
    """
    global _agent_spawn_count
    if _agent_spawn_count >= max(0, AGENT_SPAWN_MAX):
        return {"ok": False, "error": f"spawn limit reached (max {AGENT_SPAWN_MAX})"}

    name = str(args.get("name", "subagent")).strip() or "subagent"
    task = str(args.get("task", "")).strip()
    model = str(args.get("model", "")).strip() or None
    if not task:
        return {"ok": False, "error": "missing task"}

    _agent_spawn_count += 1

    system = (
        "You are a specialist sub-agent.\n"
        "You do not have file tools. You only produce reasoning and a concrete, useful output.\n"
        "Return concise Markdown. Prefer checklists, edge cases, and next actions.\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    output = call_llm(messages, model_override=model)
    return {"ok": True, "name": name, "output": output}


TOOL_IMPL = {
    "fs.list_dir": tool_fs_list_dir,
    "fs.read_file": tool_fs_read_file,
    "fs.write_file": tool_fs_write_file,
    "fs.append_file": tool_fs_append_file,
    "fs.touch": tool_fs_touch,
    "fs.search": tool_fs_search,
    "cmd.exec": tool_cmd_exec,
    "agent.spawn": tool_agent_spawn,
}


def extract_first_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None

    # 1) Raw JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) ```json fenced block
    m = re.search(r"```json\\s*(\\{.*?\\})\\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) Best-effort: find the first balanced {...} block.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def execute_tool_call(call: dict) -> dict:
    tool = str(call.get("tool", "")).strip()
    call_id = str(call.get("id", "")).strip() or None
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    started = time.time()
    ok = False
    result: dict | None = None
    err: str | None = None
    try:
        impl = TOOL_IMPL.get(tool)
        if impl is None:
            raise ValueError(f"unknown tool: {tool}")
        result = impl(args)
        ok = bool(result.get("ok", False)) if isinstance(result, dict) else True
    except Exception as exc:
        err = str(exc)
        result = {"ok": False, "error": err}
        ok = False
    elapsed = round(time.time() - started, 3)

    audit_tool(
        {
            "tool": tool,
            "id": call_id,
            "ok": ok,
            "elapsed_sec": elapsed,
        }
    )

    out = {"tool": tool, "ok": ok, "result": result, "elapsed_sec": elapsed}
    if call_id:
        out["id"] = call_id
    return out


def run_tool_loop(user_prompt: str, history: list[dict], vault_context: str, web_context: str, web_only: bool) -> str:
    allowed_cmds = ", ".join(sorted(TOOL_ALLOW_CMDS))
    system_prompt = (
        "You are a file-capable Obsidian vault assistant running inside MetaClaw.\n"
        "You can inspect and modify files using TOOLS. Use tools whenever the user asks about files, directories, "
        "searching notes, or creating/updating notes.\n\n"
        "TOOLS:\n"
        "- fs.list_dir(path, depth=1, max_entries=200)\n"
        "- fs.read_file(path, max_bytes)\n"
        "- fs.search(root, query, glob='*.md', max_files, max_matches)\n"
        "- fs.write_file(path, content, create_dirs=true)\n"
        "- fs.append_file(path, content, create_dirs=true)\n"
        "- fs.touch(path, create_dirs=true)\n"
        f"- cmd.exec(argv, cwd='/workspace', timeout_sec)  # allowed commands: {allowed_cmds}\n\n"
        "- agent.spawn(name, task, model?)  # reasoning-only sub-agent (no file tools)\n\n"
        "IMPORTANT:\n"
        "- Only use paths under /vault or /workspace.\n"
        "- Prefer fs.* tools for safety; use cmd.exec only when it is clearly simpler.\n"
        "- Keep outputs concise; do not dump huge directory trees.\n\n"
        "RESPONSE FORMAT (STRICT JSON ONLY):\n"
        "1) To call tools:\n"
        "{\"type\":\"tool_request\",\"calls\":[{\"id\":\"1\",\"tool\":\"fs.list_dir\",\"args\":{\"path\":\"/vault\",\"depth\":2}}]}\n"
        "2) To finish:\n"
        "{\"type\":\"final\",\"markdown\":\"...clean markdown...\"}\n"
    )

    agents_context = read_agents_context()
    if agents_context:
        system_prompt += "\n\n# Persistent Agent Context\n" + agents_context + "\n"

    valid_history = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        valid_history.append({"role": role, "content": content})

    user_payload = user_prompt
    if vault_context:
        user_payload += "\n\n# Retrieved Vault Context\n" + vault_context
    if web_context:
        user_payload += "\n\n# Retrieved Web Context (Tavily)\n" + web_context
    if web_only:
        user_payload += "\n\n# Mode\nWeb lookup mode requested by user; prioritize web context."

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(valid_history[-12:])
    messages.append({"role": "user", "content": user_payload})

    last_raw = ""
    for step in range(max(1, TOOL_MAX_STEPS)):
        raw = call_llm(messages)
        last_raw = raw
        obj = extract_first_json_object(raw)
        if not obj or "type" not in obj:
            # Provider/model didn't follow the JSON-only contract.
            # Give it one chance to reformat; then fall back to treating as final markdown.
            if step == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last message was not valid JSON.\n\n"
                            "Reply again using STRICT JSON ONLY (one JSON object), following the contract."
                        ),
                    }
                )
                continue
            return raw.strip()

        msg_type = str(obj.get("type", "")).strip()
        if msg_type == "final":
            return str(obj.get("markdown", "")).strip()
        if msg_type != "tool_request":
            # Unknown envelope; treat as final.
            return raw.strip()

        calls = obj.get("calls")
        if not isinstance(calls, list) or not calls:
            # Nothing to do; ask model to finish.
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "No tool calls provided. Return a final answer as JSON."})
            continue

        # Execute tool calls (cap per step).
        results = []
        for call in calls[:6]:
            if not isinstance(call, dict):
                continue
            results.append(execute_tool_call(call))

        # Feed results back.
        messages.append({"role": "assistant", "content": raw})
        tool_result_text = json.dumps(
            {"type": "tool_result", "results": results},
            ensure_ascii=False,
            indent=2,
        )
        messages.append({"role": "user", "content": tool_result_text})

    # Step limit hit.
    return (
        "I hit the maximum tool-steps limit while trying to complete your request.\n\n"
        "Here is the last model output I saw:\n\n"
        + last_raw[:800]
    )


def ensure_dirs() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def read_request() -> str:
    if not REQUEST_FILE.exists():
        return ""
    return REQUEST_FILE.read_text(encoding="utf-8", errors="ignore").strip()


def read_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        payload = json.loads(HISTORY_FILE.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(payload, list):
            return payload
    except Exception:
        pass
    return []


def read_session_scope() -> str:
    if not SESSION_FILE.exists():
        return "limited"
    try:
        payload = json.loads(SESSION_FILE.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return "limited"
    if not isinstance(payload, dict):
        return "limited"
    scope = str(payload.get("retrieval_scope", "limited")).strip().lower()
    if scope not in {"limited", "all"}:
        return "limited"
    return scope


def write_history(messages: list) -> None:
    HISTORY_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_keywords(text: str) -> list:
    tokens = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", text.lower())
    out = []
    for t in tokens:
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    # dedupe keep order
    seen = set()
    deduped = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped[:10]


def best_snippet(text: str, keywords: list) -> str:
    lower = text.lower()
    idx = -1
    for k in keywords:
        idx = lower.find(k)
        if idx >= 0:
            break
    if idx < 0:
        idx = 0
    start = max(0, idx - 220)
    end = min(len(text), idx + 420)
    snippet = text[start:end].strip()
    return re.sub(r"\s+", " ", snippet)


def retrieve_context(query: str) -> str:
    if not VAULT_DIR.exists():
        return ""

    retrieval_scope = read_session_scope()
    keywords = extract_keywords(query)
    if not keywords:
        return ""

    scored = []
    scanned = 0
    for path in VAULT_DIR.rglob("*.md"):
        # Skip hidden/system content and our own bot folder.
        rel = path.relative_to(VAULT_DIR).as_posix()
        if rel.startswith(".obsidian/") or rel.startswith(".metaclaw"):
            continue
        if retrieval_scope == "limited" and not any(rel.startswith(prefix) for prefix in LIMITED_SCOPE_PREFIXES):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lower = text.lower()
        score = sum(lower.count(k) for k in keywords)
        if score > 0:
            scored.append((score, rel, best_snippet(text, keywords)))
        scanned += 1
        if scanned >= 1200:
            break

    if not scored:
        return ""

    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:5]
    blocks = []
    for score, rel, snippet in top:
        blocks.append(f"[file: {rel} | score: {score}]\n{snippet}")
    return "\n\n".join(blocks)


def text_relevance_score(text: str, keywords: list) -> int:
    lowered = text.lower()
    return sum(lowered.count(k) for k in keywords)


def retrieve_web_context(query: str) -> str:
    if not TAVILY_API_KEY:
        return ""
    query_keywords = extract_keywords(query)
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": 4,
        "include_answer": True,
        "include_raw_content": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        TAVILY_API_URL,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

    try:
        obj = json.loads(raw)
    except Exception:
        return ""

    parts = []
    answer = obj.get("answer")
    if isinstance(answer, str) and answer.strip():
        parts.append("[tavily-answer]\n" + answer.strip())

    results = obj.get("results")
    if isinstance(results, list):
        scored_results = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()
            if not title and not content:
                continue
            score = text_relevance_score(title + " " + content, query_keywords)
            scored_results.append((score, title, url, content))
        scored_results.sort(key=lambda x: x[0], reverse=True)
        filtered = [row for row in scored_results if row[0] > 0]
        if not filtered:
            filtered = scored_results[:2]
        for score, title, url, content in filtered[:4]:
            content = re.sub(r"\s+", " ", content)[:280]
            block = f"[web] score={score} title={title or 'n/a'} url={url or 'n/a'}\n{content}"
            parts.append(block)
    return "\n\n".join(parts)


def call_openai_compatible(messages: list, *, model_override: str | None = None) -> str:
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("METACLAW_LLM_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    endpoint = base_url + "/chat/completions"

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("missing OPENAI_API_KEY/GEMINI_API_KEY in container env")

    model = (model_override or "").strip() or os.getenv("METACLAW_LLM_MODEL") or "gpt-4o-mini"

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": messages,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    def retry_sleep(attempt: int) -> None:
        base = max(0.05, LLM_RETRY_BASE_SEC) * (2 ** max(0, attempt))
        delay = min(max(0.05, base + random.uniform(0, base * 0.25)), max(0.2, LLM_RETRY_MAX_SEC))
        time.sleep(delay)

    def is_transient_http(code: int) -> bool:
        return code in {408, 425, 429, 500, 502, 503, 504}

    last_err: Exception | None = None
    for attempt in range(max(0, LLM_MAX_RETRIES) + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            last_err = None
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if attempt < LLM_MAX_RETRIES and is_transient_http(int(getattr(e, "code", 0) or 0)):
                audit_tool(
                    {
                        "tool": "llm.retry",
                        "id": None,
                        "ok": False,
                        "elapsed_sec": 0.0,
                        "detail": f"http {e.code} transient",
                    }
                )
                retry_sleep(attempt)
                last_err = e
                continue
            raise RuntimeError(f"llm http {e.code}: {body[:500]}") from e
        except Exception as e:
            # Common transient error from some providers: connection closed before a response is returned.
            msg = str(e).lower()
            transient = (
                "remote end closed connection" in msg
                or "timed out" in msg
                or "temporarily unavailable" in msg
                or "connection reset" in msg
                or "connection aborted" in msg
            )
            if attempt < LLM_MAX_RETRIES and transient:
                audit_tool(
                    {
                        "tool": "llm.retry",
                        "id": None,
                        "ok": False,
                        "elapsed_sec": 0.0,
                        "detail": f"transient error: {msg[:120]}",
                    }
                )
                retry_sleep(attempt)
                last_err = e
                continue
            raise RuntimeError(f"llm request failed: {e}") from e
    else:  # pragma: no cover
        raise RuntimeError(f"llm request failed after retries: {last_err}") from last_err

    try:
        obj = json.loads(raw)
        message = obj["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"unexpected llm response: {raw[:500]}") from e

    if isinstance(message, list):
        parts = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p).strip()

    return str(message).strip()


def llm_provider_id() -> str:
    return (os.getenv("METACLAW_LLM_PROVIDER", "") or "").strip().lower()


def call_llm(messages: list, *, model_override: str | None = None) -> str:
    provider = llm_provider_id()
    if provider == "anthropic":
        return call_anthropic(messages, model_override=model_override)
    # Default: OpenAI-compatible chat/completions (OpenAI, Gemini OpenAI endpoint, custom gateways, etc.).
    return call_openai_compatible(messages, model_override=model_override)


def call_anthropic(messages: list, *, model_override: str | None = None) -> str:
    base_url = (
        os.getenv("ANTHROPIC_BASE_URL")
        or os.getenv("METACLAW_LLM_BASE_URL")
        or "https://api.anthropic.com/v1"
    ).rstrip("/")
    endpoint = base_url + "/messages"

    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing ANTHROPIC_API_KEY in container env")

    model = (model_override or "").strip() or os.getenv("METACLAW_LLM_MODEL") or "claude-3-5-sonnet-latest"

    # Convert OpenAI-style messages into Anthropic Messages API.
    system = ""
    out_messages = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content = m.get("content", "")
        if isinstance(content, list):
            # Normalize content blocks to a plain string.
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            content = "\n".join(p for p in parts if p).strip()
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            continue
        if role == "system" and not system:
            system = content
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        out_messages.append({"role": role, "content": content})

    if not out_messages:
        out_messages = [{"role": "user", "content": "Hello"}]

    payload = {
        "model": model,
        "max_tokens": 1400,
        "temperature": 0.2,
        "messages": out_messages,
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
        },
    )

    def retry_sleep(attempt: int) -> None:
        base = max(0.05, LLM_RETRY_BASE_SEC) * (2 ** max(0, attempt))
        delay = min(max(0.05, base + random.uniform(0, base * 0.25)), max(0.2, LLM_RETRY_MAX_SEC))
        time.sleep(delay)

    def is_transient_http(code: int) -> bool:
        return code in {408, 425, 429, 500, 502, 503, 504}

    last_err: Exception | None = None
    raw = ""
    for attempt in range(max(0, LLM_MAX_RETRIES) + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            last_err = None
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if attempt < LLM_MAX_RETRIES and is_transient_http(int(getattr(e, "code", 0) or 0)):
                audit_tool(
                    {
                        "tool": "llm.retry",
                        "id": None,
                        "ok": False,
                        "elapsed_sec": 0.0,
                        "detail": f"http {e.code} transient",
                    }
                )
                retry_sleep(attempt)
                last_err = e
                continue
            raise RuntimeError(f"llm http {e.code}: {body[:500]}") from e
        except Exception as e:
            if attempt < LLM_MAX_RETRIES:
                retry_sleep(attempt)
                last_err = e
                continue
            raise RuntimeError(f"llm request failed: {e}") from e
    else:  # pragma: no cover
        raise RuntimeError(f"llm request failed after retries: {last_err}") from last_err

    try:
        obj = json.loads(raw)
        blocks = obj.get("content")
        if isinstance(blocks, list):
            parts = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
            return "\n".join(p for p in parts if p).strip()
        # Fallback
        return str(obj).strip()
    except Exception as e:
        raise RuntimeError(f"unexpected llm response: {raw[:500]}") from e


def write_response(text: str) -> None:
    RESPONSE_FILE.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    prompt = read_request()
    if not prompt:
        write_response("No prompt found in /runtime/request.txt")
        return 1

    if prompt.strip() == "/model":
        tavily_state = "enabled" if TAVILY_API_KEY else "disabled"
        write_response(
            f"configured_model={os.getenv('METACLAW_LLM_MODEL', 'unknown')}\n"
            f"tavily_web_search={tavily_state}"
        )
        return 0

    web_only = False
    web_query = prompt
    if prompt.startswith("/web "):
        web_only = True
        web_query = prompt[len("/web ") :].strip()
        if not web_query:
            write_response("usage: /web <query>")
            return 1

    history = read_history()
    context = retrieve_context(prompt)
    web_context = retrieve_web_context(web_query)

    try:
        answer = run_tool_loop(prompt, history, context, web_context, web_only)
    except Exception as e:
        write_response(f"[bot error] {e}")
        return 1

    write_response(answer)
    history.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    )
    history = history[-20:]
    write_history(history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
