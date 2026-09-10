"""Real faster-whisper run. Skipped unless faster-whisper is installed."""

from __future__ import annotations

import wave

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="pip install -e .")
pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _cpu_tiny(monkeypatch):
    """The shipped defaults are a GPU box's (cuda + large-v3-turbo). These tests
    assert on the return type only, so pin them to a CPU run of the smallest
    model -- otherwise they need a card and a 1.5 GB download."""
    from voice_to_command import core

    monkeypatch.setattr(core, "DEVICE", "cpu")
    monkeypatch.setattr(core, "COMPUTE", "int8")
    monkeypatch.setattr(core, "MODEL_SIZE", "tiny")
    monkeypatch.setattr(core, "_model", None)


def test_transcribe_samples_returns_str():
    from voice_to_command import transcribe

    silence = np.zeros(16000, dtype=np.float32)
    assert isinstance(transcribe(silence), str)


def test_transcribe_file_returns_str(tmp_path):
    from voice_to_command import transcribe

    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    assert isinstance(transcribe(str(path)), str)
