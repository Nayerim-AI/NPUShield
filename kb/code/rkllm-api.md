# RKLLM Runtime C API Reference

Target environment: Rockchip RK3588 with RKLLM Runtime and rknpu driver.

## rkllm_init

```c
int rkllm_init(RKLLMHandle *handle, RKLLMParam *param, Callback callback);
```

Load an `.rkllm` model onto the NPU and initialize the runtime handle. Blocks until initialization completes.

**Key `RKLLMParam` fields:**
- `model_path`: path to the `.rkllm` model file
- `max_context_len`: context window size (e.g. 4096 for Qwen2.5-1.5B RKLLM)
- `max_new_tokens`: maximum generation length
- `top_k`, `top_p`, `temperature`: sampling controls
- `repeat_penalty`: repetition penalty (typical: 1.1)
- `skip_special_token`: set `true` for chat output
- `is_async`: `false` for synchronous inference
- `extend_param.enabled_cpus_num`: number of CPU cores allocated to the runtime
- `extend_param.enabled_cpus_mask`: CPU affinity mask (e.g. `0xF0` for big cores 4–7)
- `extend_param.embed_flash`: set to `1` if flash embedding is supported

**Return:** `0` on success, non-zero on failure.

## rkllm_run

```c
int rkllm_run(RKLLMHandle handle, RKLLMInput *input, RKLLMInferParam *infer_param, void *userdata);
```

Run inference on a prompt.

**Input configuration:**
- `input.input_type = RKLLM_INPUT_PROMPT`
- `input.input_data.prompt_input = prompt_bytes`
- `input.role = b"user"`
- `input.enable_thinking = false` for non-thinking chat models

**Inference configuration:**
- `infer_param.mode = RKLLM_INFER_GENERATE`
- `infer_param.keep_history = 0` for stateless requests

## Callback states

- `RKLLM_RUN_NORMAL = 0`: token or chunk produced
- `RKLLM_RUN_WAITING = 1`: runtime waiting
- `RKLLM_RUN_FINISH = 2`: generation complete
- `RKLLM_RUN_ERROR = 3`: runtime error

## Prompt format for Qwen2.5 Instruct (ChatML)

```
<|im_start|>system
You are a domain-specific assistant. Answer only from the provided context.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
```

Small quantized models can hallucinate. Use RAG context and strict guardrails.

## Performance notes

- Measured throughput on RK3588 with Qwen2.5-1.5B W8A8: approximately 10–15 tokens/second.
- Actual performance depends on CPU scaling governor, memory bandwidth,
  NPU core count, and concurrent load.
- Token counters may read as `0` when the runtime does not expose per-step stats.
  Use wall-clock duration as a fallback metric.
