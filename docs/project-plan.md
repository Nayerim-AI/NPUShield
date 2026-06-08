# Project Plan

## Goal

Turn RKLLM on RK3588 NPU from experimental demo into a more production-ready local AI service.

## Original Contribution

NPUShield does not claim ownership of ezrknpu, RKLLM, RKNN Toolkit, or upstream models.

Original work:
- Production router around RKLLM
- Prompt normalization
- Output quality validator
- Retry/fallback policy
- OpenAI-compatible API wrapper
- Benchmark and quality evaluation suite
- Reproducible RK3588 deployment docs
- Before/after reports from Orange Pi 5 Pro

## Milestones

### M1 — Audit current provider
- Review `rkllm-openclaw-provider`
- Identify reusable code
- Identify state/prompt/output failure modes

### M2 — Minimal production API
- `/health`
- `/v1/models`
- `/v1/chat/completions`
- RKLLM backend adapter

### M3 — Reliability layer
- Stateless default
- Prompt template normalizer
- Output validator
- Retry policy
- Fallback adapter

### M4 — Benchmark suite
- Latency
- Tokens/sec
- RAM
- NPU load
- Repetition rate
- JSON validity
- Prompt adherence

### M5 — Arm challenge submission
- Public GitHub repo
- MIT license
- README
- Repro steps
- Benchmark report
- 3-minute demo video
