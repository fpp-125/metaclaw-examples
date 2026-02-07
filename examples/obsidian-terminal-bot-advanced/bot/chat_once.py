#!/usr/bin/env python3
import json
import os
import re
import sys
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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_API_URL = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search").strip()
LIMITED_SCOPE_PREFIXES = ("Research/", "Learning/")

STOPWORDS = {
    "the", "this", "that", "with", "from", "have", "what", "when", "where", "which", "about",
    "into", "your", "you", "and", "for", "are", "how", "can", "use", "using", "should", "please",
    "在", "这个", "那个", "怎么", "可以", "一下", "关于", "以及", "需要", "帮我", "一个",
}


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


def call_openai_compatible(messages: list) -> str:
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("METACLAW_LLM_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    endpoint = base_url + "/chat/completions"

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("missing OPENAI_API_KEY/GEMINI_API_KEY in container env")

    model = os.getenv("METACLAW_LLM_MODEL") or "gpt-4o-mini"

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

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"llm http {e.code}: {body[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"llm request failed: {e}") from e

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

    system_prompt = (
        "You are an Obsidian vault assistant running inside MetaClaw. "
        "Use vault context when relevant. Be concise, accurate, and practical. "
        "If web context is available, use it and include source URLs. "
        "If context is missing, say what to search next. "
        "Always answer in clean Markdown (headings, bullet lists, and source links)."
    )
    if web_only:
        system_prompt += " The user explicitly requested web lookup mode; prioritize web context."

    user_payload = prompt
    if context:
        user_payload += "\n\n# Retrieved Vault Context\n" + context
    if web_context:
        user_payload += "\n\n# Retrieved Web Context (Tavily)\n" + web_context

    # OpenAI-compatible chat endpoints only accept standard chat roles.
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

    history = valid_history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_payload})

    try:
        answer = call_openai_compatible(messages)
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
