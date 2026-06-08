# NPUShield — Production Layer for RKLLM on RK3588 NPU

Make Rockchip NPU LLM inference stable, measurable, and usable for real Arm edge deployments.

**Arm Create: AI Optimization Challenge 2026** — Track: Cloud AI / Physical AI

## Problem

RKLLM runs on RK3588 NPU but:
- Output quality is unstable
- Prompt format matters critically
- No production guardrails
- No quality benchmarks
- No fallback mechanism
- Not usable via standard APIs

## Solution

Production layer on top of RKLLM:
- Prompt template normalizer
- Output validator + confidence check
- Quality metrics
- Fallback to llama.cpp CPU on low confidence
- OpenAI-compatible API
- Benchmark suite
- Reproducible deployment scripts

## Structure

```
NPUShield/
├── README.md
├── LICENSE
├── src/
│   ├── api/            — OpenAI-compatible server
│   ├── core/           — Router, validator, normalizer
│   ├── providers/      — RKLLM backend, llama.cpp fallback
│   └── metrics/        — Benchmark, quality eval
├── scripts/            — Install, deploy, benchmark
├── reports/            — Generated benchmark reports
└── docs/              — Usage docs
```
