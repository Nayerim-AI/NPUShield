"""RKLLM C API provider.

Direct in-process binding to librkllmrt.so. This avoids the slow/fragile path:
FastAPI -> subprocess/CLI/ezrknpu HTTP -> RKLLM.

Target path:
FastAPI -> ctypes librkllmrt.so -> rkllm_run -> NPU.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.router import InferenceResult


RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3

RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


RKLLMHandle = ctypes.c_void_p
CallbackType = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)


DEFAULT_MODEL_PATH = os.getenv(
    "NPUSHIELD_RKLLM_MODEL",
    "models/rkllm/model.rkllm",
)
DEFAULT_LIB_PATH = os.getenv("NPUSHIELD_RKLLM_LIB", "/usr/lib/librkllmrt.so")


@dataclass
class RKLLMCAPIProvider:
    model_path: str = DEFAULT_MODEL_PATH
    lib_path: str = DEFAULT_LIB_PATH
    max_context_len: int = int(os.getenv("NPUSHIELD_RKLLM_CONTEXT", "4096"))
    max_new_tokens: int = int(os.getenv("NPUSHIELD_RKLLM_MAX_TOKENS", "1024"))
    top_k: int = int(os.getenv("NPUSHIELD_RKLLM_TOP_K", "1"))
    top_p: float = float(os.getenv("NPUSHIELD_RKLLM_TOP_P", "0.9"))
    temperature: float = float(os.getenv("NPUSHIELD_RKLLM_TEMPERATURE", "0.8"))
    repeat_penalty: float = float(os.getenv("NPUSHIELD_RKLLM_REPEAT_PENALTY", "1.1"))
    timeout_sec: int = int(os.getenv("NPUSHIELD_RKLLM_TIMEOUT", "180"))

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _lib: ctypes.CDLL | None = field(default=None, init=False, repr=False)
    _handle: RKLLMHandle = field(default_factory=RKLLMHandle, init=False, repr=False)
    _callback: CallbackType | None = field(default=None, init=False, repr=False)
    _chunks: list[str] = field(default_factory=list, init=False, repr=False)
    _state: int = field(default=-1, init=False, repr=False)
    _loaded: bool = field(default=False, init=False)
    _last_perf: dict = field(default_factory=dict, init=False)

    def name(self) -> str:
        return "rkllm-capi"

    def is_available(self) -> bool:
        return Path(self.lib_path).exists() and Path(self.model_path).exists()

    def infer(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.7) -> InferenceResult:
        if not self.is_available():
            return InferenceResult(text="", provider=self.name(), model=Path(self.model_path).name, error="rkllm_capi_unavailable")

        with self._lock:
            start = time.time()
            try:
                self._ensure_loaded(max_tokens=max_tokens, temperature=temperature)
                self._chunks.clear()
                self._state = -1
                self._last_perf.clear()

                rk_input = RKLLMInput()
                rk_input.role = b"user"
                rk_input.enable_thinking = ctypes.c_bool(False)
                rk_input.input_type = RKLLM_INPUT_PROMPT
                rk_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))

                infer_param = RKLLMInferParam()
                ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
                infer_param.mode = RKLLM_INFER_GENERATE
                infer_param.lora_params = None
                infer_param.prompt_cache_params = None
                infer_param.keep_history = 0

                ret = self._lib.rkllm_run(self._handle, ctypes.byref(rk_input), ctypes.byref(infer_param), None)
                if ret != 0:
                    return InferenceResult(text="", provider=self.name(), model=Path(self.model_path).name, error=f"rkllm_run_ret_{ret}")

                # rkllm_run is configured sync in official demo, but callback may still stream during call.
                deadline = time.time() + self.timeout_sec
                while self._state not in (RKLLM_RUN_FINISH, RKLLM_RUN_ERROR) and time.time() < deadline:
                    time.sleep(0.005)

                if self._state == RKLLM_RUN_ERROR:
                    return InferenceResult(text="".join(self._chunks), provider=self.name(), model=Path(self.model_path).name, duration_sec=time.time() - start, error="rkllm_run_error")
                if self._state != RKLLM_RUN_FINISH:
                    self.abort()
                    return InferenceResult(text="".join(self._chunks), provider=self.name(), model=Path(self.model_path).name, duration_sec=time.time() - start, error="rkllm_timeout")

                text = "".join(self._chunks).strip()
                duration = time.time() - start
                tokens = int(self._last_perf.get("generate_tokens", 0) or 0)
                gen_ms = float(self._last_perf.get("generate_time_ms", 0.0) or 0.0)
                tps = tokens / (gen_ms / 1000.0) if tokens and gen_ms else 0.0
                return InferenceResult(text=text, tokens=tokens, tokens_per_sec=tps, duration_sec=duration, provider=self.name(), model=Path(self.model_path).name)
            except Exception as exc:
                return InferenceResult(text="", provider=self.name(), model=Path(self.model_path).name, duration_sec=time.time() - start, error=str(exc))

    def abort(self) -> None:
        if self._loaded and self._lib:
            try:
                self._lib.rkllm_abort(self._handle)
            except Exception:
                pass

    def release(self) -> None:
        with self._lock:
            if self._loaded and self._lib:
                try:
                    self._lib.rkllm_destroy(self._handle)
                finally:
                    self._loaded = False

    def _ensure_loaded(self, *, max_tokens: int, temperature: float) -> None:
        if self._loaded:
            return

        self._lib = ctypes.CDLL(self.lib_path)
        self._callback = CallbackType(self._callback_impl)

        self._lib.rkllm_init.argtypes = [ctypes.POINTER(RKLLMHandle), ctypes.POINTER(RKLLMParam), CallbackType]
        self._lib.rkllm_init.restype = ctypes.c_int
        self._lib.rkllm_run.argtypes = [RKLLMHandle, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self._lib.rkllm_run.restype = ctypes.c_int
        self._lib.rkllm_abort.argtypes = [RKLLMHandle]
        self._lib.rkllm_destroy.argtypes = [RKLLMHandle]

        param = RKLLMParam()
        ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(RKLLMParam))
        param.model_path = self.model_path.encode("utf-8")
        param.max_context_len = self.max_context_len
        param.max_new_tokens = max_tokens or self.max_new_tokens
        param.skip_special_token = True
        param.n_keep = -1
        param.top_k = self.top_k
        param.top_p = ctypes.c_float(self.top_p)
        param.temperature = ctypes.c_float(temperature or self.temperature)
        param.repeat_penalty = ctypes.c_float(self.repeat_penalty)
        param.frequency_penalty = ctypes.c_float(0.0)
        param.presence_penalty = ctypes.c_float(0.0)
        param.mirostat = 0
        param.mirostat_tau = ctypes.c_float(5.0)
        param.mirostat_eta = ctypes.c_float(0.1)
        param.is_async = False
        param.img_start = b""
        param.img_end = b""
        param.img_content = b""
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        param.extend_param.enabled_cpus_num = 4
        param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

        ret = self._lib.rkllm_init(ctypes.byref(self._handle), ctypes.byref(param), self._callback)
        if ret != 0:
            raise RuntimeError(f"rkllm_init_ret_{ret}")
        self._loaded = True

    def _callback_impl(self, result, userdata, state: int) -> int:
        self._state = state
        if state == RKLLM_RUN_NORMAL and result and result.contents.text:
            self._chunks.append(result.contents.text.decode("utf-8", errors="replace"))
            try:
                perf = result.contents.perf
                self._last_perf = {
                    "prefill_time_ms": perf.prefill_time_ms,
                    "prefill_tokens": perf.prefill_tokens,
                    "generate_time_ms": perf.generate_time_ms,
                    "generate_tokens": perf.generate_tokens,
                    "memory_usage_mb": perf.memory_usage_mb,
                }
            except Exception:
                pass
        return 0
