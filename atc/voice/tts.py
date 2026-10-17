"""Piper TTS with a different voice per ATC position and a simple VHF radio effect.

Voices are downloaded on first use into models/piper/ (rhasspy/piper-voices on Hugging Face).

    python -m atc.voice.tts "Lufthansa four alpha bravo, cleared for takeoff" --pos Tower
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("tts")

DEFAULT_VOICES = {
    "Delivery": "en_US-amy-medium",
    "Ground": "en_US-ryan-medium",
    "Tower": "en_US-lessac-medium",
    "Departure": "en_GB-alan-medium",
    "Approach": "en_GB-alan-medium",
    "Center": "en_US-joe-medium",
}


def radio_effect(audio: np.ndarray, sr: int, noise: float = 0.004, seed: int = 0) -> np.ndarray:
    """Band-limit to ~300-3000 Hz, soft clip, add hiss and a squelch tail."""
    x = audio.astype(np.float32)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    spec[(f < 300) | (f > 3000)] = 0
    x = np.fft.irfft(spec, n=len(x))
    peak = float(np.max(np.abs(x))) or 1.0
    x = np.tanh(2.2 * x / peak) / np.tanh(2.2)
    rng = np.random.default_rng(seed)
    tail = int(0.12 * sr)
    x = np.concatenate([np.zeros(int(0.05 * sr), np.float32), x, np.zeros(tail, np.float32)])
    x += rng.normal(0, noise, len(x)).astype(np.float32)
    x[-tail:] += rng.normal(0, noise * 6, tail).astype(np.float32) * np.linspace(1, 0, tail)
    return (0.8 * x).astype(np.float32)


class TTS:
    def __init__(self, voices: dict[str, str] | None = None, model_dir: str | Path = "models/piper",
                 radio: bool = True, length_scale: float = 0.9):
        self.voices = {**DEFAULT_VOICES, **(voices or {})}
        self.model_dir = Path(model_dir)
        self.radio = radio
        self.length_scale = length_scale
        self._loaded: dict[str, object] = {}
        self._lock = threading.Lock()

    def _voice(self, name: str):
        if name in self._loaded:
            return self._loaded[name]
        from piper import PiperVoice
        from piper.download_voices import download_voice
        model = self.model_dir / f"{name}.onnx"
        if not model.exists():
            log.info("downloading Piper voice %s ...", name)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            download_voice(name, self.model_dir)
        v = PiperVoice.load(str(model))
        self._loaded[name] = v
        return v

    def synthesize(self, text: str, position: str = "Tower") -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig
        v = self._voice(self.voices.get(position, DEFAULT_VOICES["Tower"]))
        chunks = list(v.synthesize(text, syn_config=SynthesisConfig(length_scale=self.length_scale)))
        sr = chunks[0].sample_rate if chunks else 22050
        audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(1, np.float32)
        if self.radio:
            audio = radio_effect(audio, sr, seed=len(text))
        return audio, sr

    def speak(self, text: str, position: str = "Tower") -> None:
        """Blocking: synthesize and play. Run in an executor from asyncio."""
        import sounddevice as sd
        with self._lock:
            audio, sr = self.synthesize(text, position)
            sd.play(audio, sr)
            sd.wait()

    def preload(self) -> None:
        for name in set(self.voices.values()):
            self._voice(name)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="Speedbird one two three, runway two seven left, cleared for takeoff")
    ap.add_argument("--pos", default="Tower")
    ap.add_argument("--wav", help="write to a wav file instead of playing")
    a = ap.parse_args()
    tts = TTS()
    if a.wav:
        import wave
        audio, sr = tts.synthesize(a.text, a.pos)
        with wave.open(a.wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        print("wrote", a.wav)
    else:
        tts.speak(a.text, a.pos)
