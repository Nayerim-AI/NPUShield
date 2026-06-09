# NPUShield — Production Layer for RKLLM on RK3588 NPU

Make Rockchip NPU LLM inference stable, measurable, and usable for Arm edge deployments.

**Target:** Developers/ops working with RK3588, RKLLM, NPU edge inference.

## What It Does

```
┌─────────────────────────────────────────────────────┐
│  NPUShield                                           │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Chat API  │  │ Tool API │  │ TUI (interactive)│   │
│  └────┬─────┘  └────┬─────┘  └──────────────────┘   │
│       │              │                                │
│  ┌────▼──────────────▼────┐                           │
│  │  Persona Router         │                          │
│  │  infra  |  code          │                          │
│  └────┬──────────────┬────┘                           │
│       │              │                                  │
│  ┌────▼────┐  ┌──────▼──────┐                          │
│  │ Tool    │  │ RAG KB      │                          │
│  │ Executor│  │ (FTS5)      │                          │
│  └────┬────┘  └──────┬──────┘                          │
│       │              │                                  │
│  ┌────▼──────────────▼────┐                           │
│  │  RKLLM (1.5B NPU)      │                          │
│  │  or fallback            │                          │
│  └─────────────────────────┘                           │
└─────────────────────────────────────────────────────┘
```

## Features

### 1. Deterministic Tool Executor

Infrastructure queries → allowlist tool → live command output → direct response (skip LLM).

Tools | Description | Safe? | LLM?
---|---|---
`server_status_top` | uptime, CPU, mem, disk, top processes | ✅ safe | 🚫 skip
`service_status` | check Docker/system services | ✅ safe | 🚫 skip
`docker_ps` | list running containers with ports | ✅ safe | 🚫 skip
`safe_restart_service` | restart allowlisted Docker service | ⚠️ confirm | 🚫 skip

```
User: "check top status server"
→ tool: server_status_top
→ command: uptime + free + df + top
→ response: live server status (no LLM, ~0.2s)
```

### 2. Persona-Routed RAG

| Persona | KB | When |
|---|---|---|
| `infra` | server ops FAQ, commands | restart/status/disk/service queries |
| `code` | RKLLM API, RNKKN API, errors, cookbook | "how to init", "error X", "convert model" |

### 3. Interactive TUI

```
$ npushield-chat

npushield> /health
API healthy  model: rkllm-capi

npushield> /tools
  server_status_top — safe
  service_status    — safe
  docker_ps         — safe
  safe_restart_service — with-confirmation

npushield> check top status server
  tool=server_status_top  persona=infra
  (live uptime, load, mem, disk, CPU)

npushield> what is rkllm_init?
  provider=rkllm-capi  persona=code  dur=25s
  (explains RKLLM init API)
```

## Quickstart

### Prerequisites

- Orange Pi 5 Pro (or any RK3588 device)
- RKLLM Runtime installed with model file (`.rkllm`)
- Python 3.10+ with venv

### Install

```bash
git clone https://github.com/Nayerim-AI/NPUShield.git
cd NPUShield
python3 -m venv .venv
.venv/bin/pip install -e .
```

### Start the API server

```bash
# Set your RKLLM model path
export NPUSHIELD_RKLLM_MODEL=/path/to/model.rkllm

# Start server
.venv/bin/python -m src.api.server
# → listening on http://0.0.0.0:18999
```

### Start the TUI

```bash
.venv/bin/npushield-chat
```

## Architecture

```
src/
├── api/
│   └── server.py              # FastAPI server — chat + tool endpoints
├── core/
│   ├── persona_router.py      # Intent → infra/code persona
│   ├── tool_registry.py       # Allowlisted tool definitions
│   ├── tool_executor.py       # Local/SSH command executor
│   ├── kb_index.py            # FTS5 RAG index
│   ├── kb_loader.py           # KB file loader
│   ├── router.py              # Multi-provider LLM router
│   ├── output_validator.py    # Output quality checks
│   └── prompt_normalizer.py   # ChatML formatting
├── providers/
│   ├── rkllm_capi.py          # RKLLM NPU inference
│   └── rkllm_warm_worker.py   # Warm worker process
└── ui/
    └── chat_tui.py            # Interactive REPL client

kb/
├── infra/                     # Homelab ops knowledge base (editable)
└── code/                      # RKLLM/RKNN API reference
```

## API Endpoints

### Chat completions (OpenAI-compatible)

```http
POST /v1/chat/completions
{
  "model": "rkllm-qwen2.5-1.5b",
  "messages": [{"role": "user", "content": "check top status server"}]
}
```

Extra response headers: `x-tool`, `x-persona`, `x-provider`, `x-tokens-per-sec`.

### Tool execution

```http
GET /v1/tools                     # list available tools

POST /v1/tools/run                # execute directly
{"tool": "server_status_top", "target": "local"}
```

### Health

```http
GET /health
```

## Tool Targets

Target | Default Host | Configure via
---|---|---
`local` | localhost | (built-in)
`gateway` | (env) | `NPUSHIELD_TOOL_TARGET_GATEWAY=user@host`
`backend` | (env) | `NPUSHIELD_TOOL_TARGET_BACKEND=user@host`

## Safety

- Commands are **never generated** by the LLM — only allowlisted
- `safe_restart_service` requires explicit confirmation (HTTP 409)
- Dangerous patterns (`rm -rf`, `reboot`, `mkfs`, `docker restart`) are rejected at the executor level
- Model output validated for hallucinated functions in code persona
- Secrets (tokens, passwords) excluded from KB context

## Limitations

### RKLLM 1.5B
- **Cannot follow complex system instructions** — ignore tool context in explanation queries
- **Pretends to know** — generates generic answers instead of using injected content
- **Hallucinates functions** — fixed by output validator rejecting unknown API calls
- **Slow** — ~25s per inference on NPU (warm), ~2s flash attention

### When to skip LLM
All infrastructure tool queries skip the model entirely. The tool executor returns raw output formatted directly. This is by design — small models are unreliable for operational tasks where deterministic output is needed. For explanation queries ("jelaskan", "explain"), the tool output is passed to the model but the 1.5B model may not use it.

### Recommended upgrade
Add a **llama.cpp 7B+ fallback** for reasoning-heavy queries. The 1.5B model is adequate for:
- Tool routing (intent → persona → tool)
- Code lookup from known KB (API reference retrieval)
- Simple rephrasing of RAG context

Not adequate for:
- Complex debugging/analysis
- Following injected tool context for explanation
- Multi-step reasoning

## License

MIT
