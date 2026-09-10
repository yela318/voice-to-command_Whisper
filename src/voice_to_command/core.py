"""Speech -> text via faster-whisper.

By default it transcribes in the language spoken -- Korean stays Korean, English
stays English -- with the source language auto-detected (pin it with V2C_LANG,
e.g. "ko"). Pass translate=True (CLI: --translate, env: V2C_TRANSLATE=1) to emit
English instead, via whisper's task="translate". Input is a file path or raw mic
samples.

Per-stage timing goes to stderr as `[timing] <stage>: <s>` lines; set
V2C_TIMING=0 to silence them.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy as np

# faster-whisper model size. Must be multilingual: no ".en", and no "distil-*"
# either -- those are English-only despite the name. Use "small" on a CPU-only
# box; large-v3-turbo is the slowest option there.
MODEL_SIZE = os.environ.get("V2C_MODEL", "large-v3-turbo")
# GPU if there is one, CPU otherwise -- _load() resolves "auto" and also falls
# back when a card is present but unusable. Force it with V2C_DEVICE=cuda|cpu.
DEVICE = os.environ.get("V2C_DEVICE") or "auto"  # auto | cuda | cpu
# int8 halves the weights so the model fits on a card shared with another
# service. V2C_COMPUTE=auto -> _load() asks CTranslate2 what the card supports
# and takes float16 where offered; any other value is forced through as-is.
COMPUTE = os.environ.get("V2C_COMPUTE") or "int8"
if COMPUTE == "auto":
    COMPUTE = None
# Source language. None -> whisper auto-detects per clip. Set V2C_LANG="ko" to
# pin it for Korean-only use.
LANG = os.environ.get("V2C_LANG") or None

_model = None
_FALSEY = {"0", "false", "no", "off", ""}
# Translate the speech to English (task="translate"). Off by default -> the
# output stays in the spoken language. V2C_TRANSLATE=1 / --translate turns it on.
TRANSLATE = (os.environ.get("V2C_TRANSLATE") or "").strip().lower() not in _FALSEY


def _task(translate) -> str:
    return "translate" if (TRANSLATE if translate is None else translate) else "transcribe"


def _log(stage: str, seconds: float, note: str = "") -> None:
    if os.environ.get("V2C_TIMING", "1").strip().lower() not in _FALSEY:
        tail = " {}".format(note) if note else ""
        print("[timing] {}: {:.2f}s{}".format(stage, seconds, tail), file=sys.stderr, flush=True)


def _compute_type(device: str) -> str:
    """Best compute type CTranslate2 will actually accept here. float16 where the
    card does it efficiently (Ada/Blackwell), int8 on CPU and on older cards --
    CT2 raises ValueError for float16 below compute capability 7.0 (Pascal).
    """
    import ctranslate2

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    supported = ctranslate2.get_supported_compute_types(device)
    for candidate in ("float16", "int8"):
        if candidate in supported:
            return candidate
    return "float32"


def _preload_cuda_libs() -> None:
    """Make the pip-installed CUDA runtime findable without LD_LIBRARY_PATH.

    CTranslate2 dlopen()s `libcublas.so.12` / `libcudnn*.so.9` by soname, but the
    `nvidia-*-cu12` wheels drop them inside site-packages, where the loader does
    not look -- and LD_LIBRARY_PATH only counts if it was set before the process
    started. Loading them here with RTLD_GLOBAL means the later dlopen finds them
    already resident. Best effort: anything missing just stays missing.
    """
    import ctypes
    import glob
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None:
        return
    for root in spec.submodule_search_locations or ():
        for so in sorted(glob.glob(os.path.join(root, "*", "lib", "*.so.*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _load():
    global _model, DEVICE, COMPUTE
    if _model is None:
        if DEVICE != "cpu":
            _preload_cuda_libs()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper is required: pip install -e .") from exc
        import ctranslate2

        if DEVICE == "auto":
            DEVICE = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        COMPUTE = COMPUTE or _compute_type(DEVICE)
        try:
            _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
        except (RuntimeError, ValueError) as exc:
            # A card that is there but unusable -- driver older than the CUDA
            # runtime, missing libs, a compute type it will not take. Falling
            # back beats refusing to start, but say so: MODEL_SIZE is picked for
            # a GPU and is the slowest option on a CPU.
            if DEVICE == "cpu":
                raise
            print("v2c: {} on cuda ({}) -> falling back to cpu; {} is slow there,"
                  " consider V2C_MODEL=small".format(type(exc).__name__,
                                                     str(exc).splitlines()[0][:70], MODEL_SIZE),
                  file=sys.stderr, flush=True)
            DEVICE = "cpu"
            if COMPUTE not in ctranslate2.get_supported_compute_types("cpu"):
                COMPUTE = _compute_type("cpu")
            _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
    return _model


def warmup(translate: bool | None = None) -> None:
    """Load the model and run one throwaway inference, so the first real
    `transcribe()` pays only infer time. Call once at process start; then keep
    the process alive and call `transcribe()` back-to-back.
    """
    import numpy as np

    t0 = time.perf_counter()
    model = _load()
    # vad_filter=False on purpose: VAD drops a silent clip entirely, no segment
    # is produced and the encoder never runs -- which hides a broken CUDA setup
    # until the first real recording. Consume the generator so it actually runs.
    segments, _ = model.transcribe(
        np.zeros(16000, dtype=np.float32), language=LANG, task=_task(translate), vad_filter=False
    )
    list(segments)
    _log("warmup", time.perf_counter() - t0, "({}/{})".format(DEVICE, COMPUTE))


def transcribe(audio: Union[str, "np.ndarray"], translate: bool | None = None) -> str:
    """Speech -> text.

    `audio` is an audio file path (wav/m4a/mp3/... -- faster-whisper decodes it)
    or 16 kHz mono float32 samples (as returned by `record()`). Returns text in
    the spoken language; pass `translate=True` to translate it to English.
    `translate=None` follows the V2C_TRANSLATE env var (default: off).
    """
    cold = _model is None
    t0 = time.perf_counter()
    model = _load()
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    segments, _ = model.transcribe(
        audio,
        language=LANG,
        task=_task(translate),
        vad_filter=True,  # trims silence, cuts hallucination on short clips
        condition_on_previous_text=False,
    )
    # segments is lazy -- the real work (decode for files + VAD + inference)
    # happens while it is consumed here.
    text = " ".join(s.text.strip() for s in segments).strip()
    t_infer = time.perf_counter() - t1

    _log("model_load", t_load, "({}, {}/{})".format(
        "cold start" if cold else "cached", DEVICE, COMPUTE))
    _log("infer", t_infer)
    _log("total", t_load + t_infer)
    return text
