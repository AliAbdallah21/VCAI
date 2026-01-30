# tts/agent.py
import os
import numpy as np

from shared.constants import TTS_DEFAULT_VOICE
from tts.chatterbox_model import ChatterboxTTSModel, ChatterboxTTSConfig

# Singleton model (load once per process)
_tts_model: ChatterboxTTSModel | None = None


def _get_model() -> ChatterboxTTSModel:
    global _tts_model
    if _tts_model is None:
        cfg = ChatterboxTTSConfig(
            device="cuda",
            default_language_id="ar",
        )
        # Optional: set HF token if needed
        # os.environ["HF_TOKEN"] = "hf_..."
        _tts_model = ChatterboxTTSModel(cfg)
        _tts_model.load()
    return _tts_model


def text_to_speech(text: str, voice_id: str = "default", emotion: str = "neutral") -> np.ndarray:
    """
    Project interface wrapper.
    - voice_id maps to an audio_prompt_path (voice cloning reference)
    - emotion maps to exaggeration/cfg_weight presets (simple mapping)
    """
    model = _get_model()

    # Map voice_id -> reference wav path (you decide your folder structure)
    audio_prompt_path = None
    if voice_id and voice_id != "default":
        # Example: data/voices/egyptian_male_01.wav
        candidate = f"data/voices/{voice_id}.wav"
        if os.path.exists(candidate):
            audio_prompt_path = candidate

    # Simple emotion → parameters mapping (you can tune later)
    emotion_presets = {
        "neutral": dict(exaggeration=0.5, cfg_weight=0.5),
        "friendly": dict(exaggeration=0.6, cfg_weight=0.55),
        "happy": dict(exaggeration=0.7, cfg_weight=0.6),
        "frustrated": dict(exaggeration=0.8, cfg_weight=0.65),
        "interested": dict(exaggeration=0.6, cfg_weight=0.6),
        "hesitant": dict(exaggeration=0.45, cfg_weight=0.5),
    }
    preset = emotion_presets.get(emotion, emotion_presets["neutral"])

    wav = model.synthesize(
        text=text,
        language_id="ar",  # your STT_LANGUAGE default is ar
        audio_prompt_path=audio_prompt_path,
        **preset,
    )
    return wav
