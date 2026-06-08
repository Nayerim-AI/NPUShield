# RK3588/RKLLM/RKNN Error Notes

## Warning: rknpu driver version is too low

Example:
```text
W rkllm: Warning: Your rknpu driver version is too low, please upgrade to 0.9.7
I rkllm: rkllm-runtime version: 1.2.1, rknpu driver version: 0.9.6, platform: RK3588
```

Meaning:
- Runtime works, but driver is older than recommended.
- Warning is not necessarily fatal.
- Upgrade target is rknpu driver 0.9.7+.

Action:
- Continue testing if inference succeeds.
- Upgrade driver if instability, poor performance, or runtime incompatibility appears.

## Model gives wrong factual answer

Cause:
- Small model (Qwen2.5 1.5B) + W8A8 quantization can hallucinate.
- Knowledge may be degraded.

Fix:
- Use RAG context.
- Add strict domain-specific system prompt.
- Add output validator that rejects answers not grounded in context.
- Use fallback model for high-risk factual tasks.

## rkllm_init fails

Common causes:
- Wrong model path.
- `.rkllm` model built for incompatible runtime version.
- NPU driver not loaded.
- Not enough memory.
- Wrong `librkllmrt.so` path.

## Low tokens/sec or zero token metrics

Common causes:
- Runtime callback did not expose perf stats.
- The wrapper failed to parse performance counters.
- Model is generating but token counters are unavailable.

Action:
- Use wall-clock `duration_sec` as fallback metric.
- Inspect callback `perf.generate_tokens` and `perf.generate_time_ms`.
