# RKLLM/RK3588 Cookbook

## Minimal C API inference flow

1. Fill `RKLLMParam` with model path, context length, sampling config.
2. Call `rkllm_init` and check return code.
3. Build ChatML prompt.
4. Fill `RKLLMInput` with `input_type = RKLLM_INPUT_PROMPT`.
5. Fill `RKLLMInferParam` with `mode = RKLLM_INFER_GENERATE` and `keep_history = 0`.
6. Call `rkllm_run`.
7. Collect chunks in callback until `RKLLM_RUN_FINISH`.
8. Clean special tokens.
9. Validate output against task/persona guardrail.

## Recommended NPUShield approach

For tiny RKLLM models, do not use the model as a general knowledge source.
Use:

```text
persona router -> FTS5 RAG -> strict prompt -> RKLLM C API -> validator
```

## When to use RAG

Use RAG for:
- homelab/infra answers
- RKLLM/RKNN API help
- customer service
- technical teaching

RAG is optional for:
- translation
- paraphrase
- pantun/creative short text

## Safe prompt for code helper

```text
Jawab hanya dari konteks dokumentasi berikut.
Jika konteks tidak memuat jawaban, katakan: "Saya belum punya info itu di knowledge base."
Jangan membuat nama API, parameter, atau command yang tidak ada di konteks.
```

## Safe prompt for infra helper

```text
Jawab hanya dari inventory server, services, FAQ, dan command allowlist.
Jangan tampilkan secret/token/password.
Untuk command destructive, minta konfirmasi dulu.
Jika info tidak ada, katakan belum ada di knowledge base.
```
