# NPUShield — Knowledge Base and Persona Design

## Target Market

NPUShield is not a general-purpose chatbot.

It targets developers and operators working with:

- ARM64 edge devices
- Rockchip RK3588 boards
- RKLLM Runtime
- RKNN Toolkit2
- small quantized local LLMs
- self-hosted inference and homelab operations

The product goal is:

```text
A guarded, domain-specific RAG layer for tiny NPU-hosted language models.
```

The assistant should not pretend that a 1.5B quantized model is a reliable general knowledge engine. Instead, it should answer from curated knowledge bases, strict persona prompts, and validation rules.

---

## Persona 1 — Infra Assistant

### Job

Help operators reason about Linux servers, ARM boards, Docker services, private networks, reverse proxies, and safe maintenance commands.

### Knowledge Base Layout

```text
kb/infra/
├── servers.yaml      # Generic server inventory template
├── services.yaml     # Generic service registry template
├── faq.md            # Common operations and safe procedures
└── commands.yaml     # Allowlisted commands and dangerous patterns
```

### What belongs here

- generic server inventory templates
- example safe commands
- reverse proxy concepts
- service health-check patterns
- Docker status commands
- restart procedures with confirmation rules

### What must not be committed here

- real public IP addresses
- customer domains
- SSH usernames tied to a real deployment
- Cloudflare/GitHub/Gitea/API tokens
- private keys
- passwords
- internal service URLs from a live infrastructure

Real deployment data should live in a private overlay, not in the public repository.

### Guardrails

- If a command is destructive, ask for explicit confirmation.
- Never print secrets or credentials.
- If the KB does not contain the answer, say:

```text
I do not have that information in the knowledge base.
```

- Do not invent hostnames, IP addresses, service names, or operational status.

---

## Persona 2 — Code Helper for ARM/RKLLM/RKNN

### Job

Help developers working with Rockchip NPU software stacks: RKLLM C API, RKNN Toolkit2, driver warnings, ctypes wrappers, FastAPI inference services, and prompt formatting.

### Knowledge Base Layout

```text
kb/code/
├── rkllm-api.md      # RKLLM Runtime C API notes
├── rknn-api.md       # RKNN Toolkit2 notes (future)
├── rknpu-driver.md   # Driver/version troubleshooting (future)
├── cookbook.md       # Common workflows
├── errors.md         # Known errors and fixes
└── snippets/         # Tested code snippets
```

### Guardrails

- Do not invent APIs, structs, parameters, or function names.
- Always answer from retrieved context when in strict mode.
- Mention runtime/API version assumptions when relevant.
- If the code example is not in context, say that the KB does not contain a tested example.

---

## Architecture

```text
User Query
   ↓
Persona Router
   ↓
Persona-specific FTS5 retrieval
   ↓
Context Builder
   ↓
RKLLM C API using ChatML
   ↓
Output Validator
   ↓
Response with metadata
```

### Components

| Component | Technology | Reason |
|---|---|---|
| Retriever | SQLite FTS5 | Lightweight, built-in, no embedding model required |
| KB format | Markdown + YAML | Editable, diffable, Git-friendly |
| Persona router | Keyword/regex rules | Deterministic and easy to debug |
| Context injection | Top-N retrieved docs | Fits 4K context windows |
| Output validation | Generic + persona-aware checks | Reduces unsafe commands and hallucinated APIs |

---

## Why FTS5 first?

Embedding search is useful, but it adds memory, CPU, and dependency overhead. On ARM edge devices, simple lexical retrieval is often enough for:

- command lookup
- error message lookup
- API name lookup
- FAQ matching
- service registry lookup

NPUShield can later add an embedding backend, but FTS5 is the correct first step for an RK3588-friendly public demo.

---

## Public vs Private KB

The public repository should include only reusable templates and generic technical references.

Recommended structure:

```text
kb/
├── infra/          # public templates and generic procedures
├── code/           # public RKLLM/RKNN notes
└── overlays/       # ignored/private in real deployments
```

Private overlays can contain real inventory and service data, but they must be excluded from GitHub.

Suggested `.gitignore` rule:

```text
kb/overlays/
*.private.yaml
*.secrets.yaml
```

---

## Response metadata

The OpenAI-compatible API response includes debug metadata:

```json
{
  "x-persona": "infra",
  "x-rag-docs": 3,
  "x-rag-paths": ["infra/faq.md", "infra/commands.yaml"],
  "x-safety-flags": []
}
```

This makes demos easier to audit: the user can see which persona and KB files were used.

---

## Product positioning

```text
NPUShield: Guarded RAG for tiny local LLMs on Rockchip NPU devices.
```

The value is not that the tiny model knows everything.

The value is that NPUShield turns a tiny local model into a safer domain assistant by adding:

- persona routing
- curated knowledge
- lightweight retrieval
- strict prompts
- output validation
- optional fallback models
