from src.core.output_validator import OutputValidator
from src.core.prompt_normalizer import PromptNormalizer


def test_prompt_normalizer_uses_qwen_chatml():
    prompt = PromptNormalizer().normalize([
        {"role": "user", "content": "Apa itu NPU?"},
    ])
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "Apa itu NPU?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_validator_rejects_empty_output():
    result = OutputValidator().validate("")
    assert not result.ok
    assert "empty_output" in result.reasons


def test_validator_rejects_runtime_noise():
    result = OutputValidator().validate("Jawaban [Token/s]: 12.3 [Tokens]: 10")
    assert not result.ok or "runtime_noise" in result.reasons


def test_validator_accepts_normal_answer():
    result = OutputValidator().validate("NPU adalah prosesor khusus untuk mempercepat operasi AI di perangkat edge.")
    assert result.ok
    assert result.score >= 0.65
