from src.providers.rkllm_stateless import StatelessRKLLMProvider


def test_stateless_clean_prefers_assistant_marker():
    raw = "Welcome to ezrkllm\n<|im_start|>assistant\nNPU adalah akselerator AI.\n[Token/s]: 10.0\n[Tokens]: 5\nYou:"
    assert StatelessRKLLMProvider._clean(raw) == "NPU adalah akselerator AI."


def test_stateless_unavailable_on_missing_binary():
    p = StatelessRKLLMProvider(binary_path="/missing/rkllm", model_path="/missing/model.rkllm")
    assert not p.is_available()
