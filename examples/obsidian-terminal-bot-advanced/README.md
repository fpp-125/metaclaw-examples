# Obsidian Terminal Bot (Advanced, Image Mode)

A full-featured MetaClaw example with:

- business logic baked into the image layer (`bot/chat_once.py` + `image/Dockerfile`)
- `agent.claw` kept as runtime policy/config/entrypoint only
- host-side TUI (`chat_tui.py`) for safer write flow and better UX

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

## Run Chat

```bash
export GEMINI_API_KEY='...'
export TAVILY_API_KEY='...'
./chat.sh
```

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
- `/history`
- `/render` or `/render plain|glow|demo`
- `/net` or `/net none|out [--default]`
- `/scope` or `/scope limited|all [--default]`
- `/confirm` or `/confirm once|diff|auto [--default]`
- `/save <Research/.../file.md>`
- `/save --default-dir <Research/...> [--default]`
- `/reset`
- `/model`
- `/web <query>`
- `/clear`
- `/exit`

Command menu keys:

- `h/l` move focus
- `space` or `Enter` select
- `q`/`ESC` cancel

## Security Notes

- Default network is `none`; use `/net out` only when needed.
- Vault is mounted read-only inside container.
- Save to vault happens on host side via `/save`, with scope limits and confirmation.
- API keys are runtime-injected (`--llm-api-key-env`, `--secret-env`), not committed into files.
