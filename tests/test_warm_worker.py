from src.providers.rkllm_warm_worker import WarmRKLLMWorker


def test_warm_worker_clean_prefers_assistant_marker():
    raw = "banner\n<|im_start|>assistant\nNPU adalah akselerator AI.\n[Token/s]: 10.0\n[Tokens]: 5\nYou:"
    assert WarmRKLLMWorker._clean(raw) == "NPU adalah akselerator AI."


def test_warm_worker_should_recycle_without_child():
    worker = WarmRKLLMWorker(binary_path="/missing/rkllm", model_path="/missing/model.rkllm")
    assert worker._should_recycle()


def test_warm_worker_status_shape():
    worker = WarmRKLLMWorker(binary_path="/missing/rkllm", model_path="/missing/model.rkllm")
    status = worker.status()
    assert status["provider"] == "rkllm-warm-worker"
    assert status["alive"] is False
