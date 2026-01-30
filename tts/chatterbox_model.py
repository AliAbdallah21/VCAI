# tts/chatterbox_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

import numpy as np
import torch
import torchaudio as ta

from shared.constants import TTS_SAMPLE_RATE

# IMPORTANT: this is the multilingual class that worked for you
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


@dataclass
class ChatterboxTTSConfig:
    device: str = "cuda"
    default_language_id: str = "ar"   # your project default
    sample_rate_out: int = TTS_SAMPLE_RATE  # 22050 or 24000
    # optional: HF token if repo is gated in some environments
    hf_token_env: str = "HF_TOKEN"


class ChatterboxTTSModel:
    """
    Wraps Chatterbox Multilingual TTS.
    - Loads once
    - generate(text, language_id, audio_prompt_path=...) -> np.float32 waveform
    """

    def __init__(self, cfg: ChatterboxTTSConfig):
        self.cfg = cfg
        self.device = self._pick_device(cfg.device)
        self._model = None
        self._model_sr = None

    def _pick_device(self, preferred: str) -> str:
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self) -> None:
        """Load model once."""
        # If you ever need auth, set HF_TOKEN in env before load
        # os.environ["HF_TOKEN"] = "hf_..."
        token = os.getenv(self.cfg.hf_token_env)
        if token:
            os.environ["HF_TOKEN"] = token

        self._model = ChatterboxMultilingualTTS.from_pretrained(self.device)
        # From your logs: model.sr exists and was 24000
        self._model_sr = getattr(self._model, "sr", 24000)

    @property
    def sr(self) -> int:
        return int(self._model_sr or 24000)

    def synthesize(
        self,
        text: str,
        language_id: Optional[str] = None,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
    ) -> np.ndarray:
        """
        Returns: np.ndarray float32, shape (n_samples,), sample rate = cfg.sample_rate_out
        """
        if self._model is None:
            self.load()

        language_id = language_id or self.cfg.default_language_id

        # Your signature: generate(text, language_id, audio_prompt_path=None, ...)
        wav_torch = self._model.generate(
            text,
            language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

        # Ensure CPU float32
        if isinstance(wav_torch, torch.Tensor):
            wav = wav_torch.detach().float().cpu()
        else:
            wav = torch.tensor(wav_torch).float().cpu()

        # Resample if project expects 22050 but model is 24000
        if self.cfg.sample_rate_out and self.cfg.sample_rate_out != self.sr:
            wav = ta.functional.resample(wav, self.sr, self.cfg.sample_rate_out)

        # Return 1D numpy float32
        wav_np = wav.numpy().astype(np.float32)
        return wav_np
