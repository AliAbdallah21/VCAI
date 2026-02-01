# tts/agent.py
"""
TTS Agent using Chatterbox Multilingual with Egyptian fine-tuned checkpoint.
Loads base model first, then merges fine-tuned T3 weights.
"""
import os
import numpy as np

from shared.constants import TTS_DEFAULT_VOICE
from tts.chatterbox_model import ChatterboxTTSModel, ChatterboxTTSConfig

# Singleton model (load once per process)
_tts_model: ChatterboxTTSModel | None = None

# ══════════════════════════════════════════════════════════════════════════════
# EGYPTIAN FINE-TUNE CHECKPOINT
# Set to None to use base model only
# ══════════════════════════════════════════════════════════════════════════════
EGYPTIAN_CHECKPOINT = r"C:\chatterboxMulti\egyptian-finetune\output_dahih_shaghal\checkpoint-2000\model.safetensors"


def _get_model() -> ChatterboxTTSModel:
    global _tts_model
    if _tts_model is None:
        cfg = ChatterboxTTSConfig(
            device="cuda",
            default_language_id="ar",
        )
        _tts_model = ChatterboxTTSModel(cfg)
        _tts_model.load()

        # ══════════════════════════════════════════════════════════════════
        # LOAD EGYPTIAN FINE-TUNED WEIGHTS ON TOP OF BASE MODEL
        # ══════════════════════════════════════════════════════════════════
        if EGYPTIAN_CHECKPOINT and os.path.exists(EGYPTIAN_CHECKPOINT):
            try:
                from safetensors.torch import load_file
                print(f"[TTS] Loading Egyptian fine-tuned checkpoint...")
                print(f"[TTS] Path: {EGYPTIAN_CHECKPOINT}")

                weights = load_file(EGYPTIAN_CHECKPOINT, device=cfg.device)
                missing, unexpected = _tts_model._model.t3.load_state_dict(weights, strict=False)

                print(f"[TTS] ✅ Egyptian checkpoint loaded: {len(weights)} parameters")
                if missing:
                    print(f"[TTS]    Missing keys: {len(missing)} (expected - frozen layers)")
                if unexpected:
                    print(f"[TTS]    Unexpected keys: {len(unexpected)}")

                # Set T3 to eval mode after loading
                _tts_model._model.t3.eval()
                print(f"[TTS] ✅ Model ready with Egyptian dialect!")

            except Exception as e:
                print(f"[TTS] ⚠️ Failed to load checkpoint: {e}")
                print(f"[TTS] Continuing with base model...")
        else:
            if EGYPTIAN_CHECKPOINT:
                print(f"[TTS] ⚠️ Checkpoint not found: {EGYPTIAN_CHECKPOINT}")
                print(f"[TTS] Using base model instead")

    return _tts_model


def text_to_speech(text: str, voice_id: str = "default", emotion: str = "neutral", language_id: str = "ar") -> np.ndarray:
    """
    Project interface wrapper.
    - voice_id maps to an audio_prompt_path (voice cloning reference)
    - emotion maps to exaggeration/cfg_weight presets
    """
    model = _get_model()

    # Map voice_id -> reference wav path
    audio_prompt_path = None
    if voice_id and voice_id != "default":
        candidate = f"data/voices/{voice_id}.wav"
        if os.path.exists(candidate):
            audio_prompt_path = candidate

    # Emotion → parameters mapping
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
        language_id=language_id,
        audio_prompt_path=audio_prompt_path,
        **preset,
    )
    return wav.flatten()