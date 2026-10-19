"""Microphone recording + faster-whisper transcription.

The Whisper model is downloaded on first use (Hugging Face, Systran/faster-whisper-*).

    python -m atc.voice.stt                 # hold Enter-to-stop recording test
    python -m atc.voice.stt --file x.wav    # transcribe a file
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("stt")

SR = 16000


class Recorder:
    """Records from the default input device between start() and stop()."""

    def __init__(self, device: int | str | None = None):
        self.device = device
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self) -> None:
        import sounddevice as sd
        with self._lock:
            if self._stream is not None:
                return
            self._chunks = []
            self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                          device=self.device, callback=self._cb)
            self._stream.start()

    def _cb(self, indata, frames, t, status):  # noqa: ARG002
        self._chunks.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray:
        with self._lock:
            if self._stream is None:
                return np.zeros(0, np.float32)
            self._stream.stop()
            self._stream.close()
            self._stream = None
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)

    @property
    def active(self) -> bool:
        return self._stream is not None


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    n = int(round(x.size * sr_out / sr_in))
    return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    """16-bit PCM wav -> mono float32 at 16 kHz (avoids PyAV for the CLI test)."""
    import wave
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample(x, sr)


def _add_cuda_dll_dirs() -> None:
    """Make pip-installed CUDA libs (nvidia-cublas-cu12, nvidia-cudnn-cu12) loadable on Windows."""
    import os
    import site
    import sys
    if sys.platform != "win32":
        return
    for sp in site.getsitepackages():
        base = os.path.join(sp, "nvidia")
        if not os.path.isdir(base):
            continue
        for sub in os.listdir(base):
            b = os.path.join(base, sub, "bin")
            if os.path.isdir(b):
                os.add_dll_directory(b)
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")


class STT:
    def __init__(self, model: str = "small.en", device: str = "auto", compute_type: str = "auto",
                 download_root: str = "models/whisper"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._model = None

    def load(self):
        if self._model is None:
            if self.device != "cpu":
                _add_cuda_dll_dirs()
            log.info("loading whisper %s (%s)...", self.model_name, self.device)
            self._model = self._create(self.device, self.compute_type)
        return self._model

    def _create(self, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        def make(dev, ct, local_only):
            return WhisperModel(self.model_name, device=dev, compute_type=ct,
                                download_root=self.download_root, local_files_only=local_only)

        for dev, ct in ((device, compute_type), ("cpu", "int8")):
            try:
                # cached copy first: the online revision check costs ~30 s through a VPN
                try:
                    return make(dev, ct, True)
                except (FileNotFoundError, ValueError, OSError):
                    return make(dev, ct, False)
            except Exception as e:  # noqa: BLE001 - e.g. CUDA libs missing -> CPU
                if dev == "cpu":
                    raise
                log.warning("whisper on %s failed (%s), using CPU int8", dev, e)
        raise RuntimeError("unreachable")

    def _run(self, audio: np.ndarray, prompt: str) -> str:
        segments, _ = self.load().transcribe(audio, language="en", beam_size=5, initial_prompt=prompt or None,
                                             vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> str:
        if audio.size < SR * 0.3:
            return ""
        try:
            text = self._run(audio, prompt)
        except RuntimeError as e:
            if self.device == "cpu":
                raise
            # CUDA runtime (cuBLAS/cuDNN) missing: CTranslate2 only fails at first encode
            log.warning("whisper GPU failed (%s); switching to CPU int8", e)
            self.device, self.compute_type, self._model = "cpu", "int8", None
            text = self._run(audio, prompt)
        log.info("STT: %s", text)
        return text


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--prompt", default="")
    a = ap.parse_args()
    stt = STT(a.model)
    if a.file:
        print(stt.transcribe(load_wav(a.file), a.prompt))
    else:
        rec = Recorder()
        input("Press Enter to start recording...")
        rec.start()
        input("Recording. Press Enter to stop...")
        print(stt.transcribe(rec.stop(), a.prompt))
