# Obsidian Terminal Bot (Advanced, Image Mode)

A full-featured MetaClaw example with:

- business logic baked into the image layer (`bot/chat_once.py` + `image/Dockerfile`)
- `agent.claw` kept as runtime policy/config/entrypoint only
- host-side TUI (`chat_tui.py`) for safer write flow and better UX
- in-container tools for autonomous file ops (list/read/search/write + allowlisted CLI) inside the isolated habitat

## What You Configure

Edit `agent.claw` mount sources before first run:

- `/ABS/PATH/TO/OBSIDIAN_VAULT` -> your vault root (mounted to `/vault`, read-only)
- `/ABS/PATH/TO/BOT_HOST_DATA/*` -> host data directories outside vault (`config`, `logs`, `runtime`, `workspace`)

This separation avoids overlapping mounts and keeps runtime data outside your notes.

## Build Image

Default builder is Apple Container CLI (`container`).
Use `RUNTIME_BIN=docker` or `RUNTIME_BIN=podman` if preferred.

```bash
cd examples/obsidian-terminal-bot-advanced
./build_image.sh
```

`build_image.sh` compiles image `metaclaw/obsidian-terminal-bot:local`, resolves digest, and updates `agent.claw`.

## Upgrade (In Place)

If this project was created via `metaclaw quickstart/onboard`, it includes a template lock under `.metaclaw/`.
You can refresh managed template files (bot business code + image layer) without overwriting your `agent.claw`, `.env`, or `vault/`:

```bash
metaclaw project upgrade --project-dir=.
```

After upgrading, rebuild the image to pick up the updated `bot/` code:

```bash
RUNTIME_BIN=container ./build_image.sh
```

## Friend Setup (Engine + Advanced Bot)

```bash
# 0) Clone both repos
git clone https://github.com/fpp-125/metaclaw.git
git clone https://github.com/fpp-125/metaclaw-examples.git

# 1) Build the engine binary
cd metaclaw
go build -o ./metaclaw ./cmd/metaclaw

# 2) Enter this example
cd ../metaclaw-examples/examples/obsidian-terminal-bot-advanced

# 3) Prepare host data directories
BOT_DATA="$HOME/.metaclaw/obsidian-terminal-bot"
mkdir -p "$BOT_DATA"/{config,logs,runtime,workspace}
```

Then do two things:

- Edit `agent.claw`.
- Replace `/ABS/PATH/TO/OBSIDIAN_VAULT` with your own vault path.
- Replace `/ABS/PATH/TO/BOT_HOST_DATA` with the absolute path of `BOT_DATA`.
- Build and run:

```bash
# 4) Build image (choose one runtime)
RUNTIME_BIN=container ./build_image.sh
# RUNTIME_BIN=docker ./build_image.sh
# RUNTIME_BIN=podman ./build_image.sh

# 5) Set API keys and run
export OPENAI_FORMAT_API_KEY='...'
export TAVILY_API_KEY='...'   # optional

METACLAW_BIN="/ABS/PATH/TO/metaclaw/metaclaw" \
RUNTIME_TARGET=apple_container \
./chat.sh
# or set RUNTIME_TARGET=docker / podman
```

If `metaclaw` is already in your `PATH`, you can omit `METACLAW_BIN=...`.

## Run Chat

```bash
export OPENAI_FORMAT_API_KEY='...'
export TAVILY_API_KEY='...'
./chat.sh
```

Or, let the TUI manage keys and provider/model selection:

1. Start chat: `./chat.sh`
2. Run: `/llm setup` (multi-select providers/models, hidden key input, optional `.env` write)

Optional runtime override:

```bash
RUNTIME_TARGET=apple_container ./chat.sh
RUNTIME_TARGET=podman ./chat.sh
RUNTIME_TARGET=docker ./chat.sh
```

If `metaclaw` is not in your PATH:

```bash
METACLAW_BIN=/abs/path/to/metaclaw ./chat.sh
```

## TUI Commands

- `/help`
- `/status`
- `/llm` (show status)
- `/llm setup` (choose providers/models, update keys)
- `/llm use` (switch default model)
- `/llm key` (update keys)
- `/history`
- `/render` or `/render plain|glow|demo`
- `/focus` or `/focus start|end|stay|input [--default]`
- `/show` or `/show start|end`
- `/net` or `/net none|out [--default]`
- `/vault` or `/vault ro|rw [--default]`
- `/scope` or `/scope limited|all [--default]`
- `/confirm` or `/confirm once|diff|auto [--default]`
- `/save <Research/.../file.md>`
- `/append <Research/.../file.md>`
- `/touch <Research/.../file.md>`
- `/save --default-dir <Research/...> [--default]`
- `/reset`
- `/web <query>`
- `/clear`
- `/exit`

Command menu keys:

- `h/l` move focus
- `space` or `Enter` select
- `q`/`ESC` cancel

Output focus modes:

- `start`: page the response from the top (best for long answers)
- `end`: page the response from the bottom
- `stay`: print inline (normal scrollback)
- `input`: keep you at the prompt, store the response for `/show`

## In-Container Tools (Autonomous)

The bot process runs inside the container and can (when policy allows):

- list/read/search files under `/vault` and `/workspace`
- write/append/touch files under `/vault` and `/workspace` (requires vault mount `rw` for vault writes)
- run a small allowlisted set of commands via `cmd.exec` (default: `ls`, `find`, `grep`)
- spawn reasoning-only sub-agents via `agent.spawn` (no file tools)

Tool calls are best-effort audited to `logs/tool_audit.jsonl` (tool name + status + timing). API keys are never written to files.

## AGENTS.md + soul.md

This template includes `agents/AGENTS.md` and `agents/soul.md`.

- `agents/AGENTS.md`: operational rules and tool contract (managed by template upgrades unless you modify it)
- `agents/soul.md`: persona and style (user-owned; not overwritten by upgrades)

The host TUI syncs both into the mounted runtime dir as `/runtime/agents_context.md` on each run.

## Security Notes

- Default network is `none`; use `/net out` only when needed.
- Vault is mounted read-only inside container by default.
- Save to vault happens on host side via `/save`, with scope limits and confirmation.
- API keys are runtime-injected (`--llm-api-key-env`, `--secret-env`), not committed into files.

## Option B (Less Safe): Let The Container Write The Vault Directly

If you want the bot process inside the container to create/modify files under `/vault`, you can mount the vault as read-write.

In `agent.claw`, change the vault mount:

```yaml
    mounts:
      - source: /ABS/PATH/TO/OBSIDIAN_VAULT
        target: /vault
        readOnly: false
```

Notes:
- This example bot still defaults to host-side `/save`. Vault write access only changes what the container is *allowed* to do.
- This bypasses the host-side `/save` guardrails. Treat it as **unsafe-by-default**.
- If you enabled outbound network access, a compromised bot can both read your vault and exfiltrate it.
- Mitigations: keep backups, prefer mounting a smaller subfolder instead of the whole vault, and keep network `none` unless you explicitly need it.

If you created this project via engine quickstart/onboard, you can enable this mode up front with:

```bash
metaclaw quickstart obsidian --vault-write ...
metaclaw onboard obsidian --vault-write ...
```
