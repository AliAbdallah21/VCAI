# tts/agent.py
"""
TTS Agent using Chatterbox Multilingual with Egyptian fine-tuned checkpoint.
Loads base model first, then merges fine-tuned T3 weights.
"""
import logging
import os
import random
import numpy as np
import torch

from shared.constants import TTS_DEFAULT_VOICE
from tts.chatterbox_model import ChatterboxTTSModel, ChatterboxTTSConfig

_log = logging.getLogger(__name__)

# Singleton model (load once per process)
_tts_model: ChatterboxTTSModel | None = None
# Set to True when the Egyptian fine-tuned checkpoint loads successfully
_egyptian_active: bool = False

# ══════════════════════════════════════════════════════════════════════════════
# EGYPTIAN FINE-TUNE CHECKPOINT
# Set to None to use base model only
# ══════════════════════════════════════════════════════════════════════════════
EGYPTIAN_CHECKPOINT = r"C:\chatterboxMulti\egyptian-finetune\output_audiobooks_multispeaker_2\final_model\model.safetensors"

# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK PHRASES
# Used when synthesis itself fails — the virtual customer naturally asks the
# salesperson to repeat themselves, keeping the conversation alive.
# ══════════════════════════════════════════════════════════════════════════════
_FALLBACK_PHRASES = [
    "معلش؟ ممكن تعيد تاني؟ صوتك مش واضح",
    "مش سامعك كويس، ممكن تتكلم أعلى شوية؟",
    "معلش، في مشكلة في الاتصال، ممكن تعيد؟",
    "صوتك بيتقطع، ممكن تقول تاني؟",
]


def _get_model() -> ChatterboxTTSModel:
    global _tts_model, _egyptian_active
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

                print(f"[TTS] Egyptian checkpoint loaded: {len(weights)} parameters")
                if missing:
                    print(f"[TTS]    Missing keys: {len(missing)} (expected - frozen layers)")
                if unexpected:
                    print(f"[TTS]    Unexpected keys: {len(unexpected)}")

                _tts_model._model.t3.eval()
                _egyptian_active = True
                print(f"[TTS] Model ready with Egyptian dialect!")

            except Exception as e:
                print(f"[TTS] Warning: Failed to load checkpoint: {e}")
                print(f"[TTS] Continuing with base model...")
        else:
            if EGYPTIAN_CHECKPOINT:
                print(f"[TTS] Warning: Checkpoint not found: {EGYPTIAN_CHECKPOINT}")
                print(f"[TTS] Using base model instead")

    return _tts_model


def text_to_speech(
    text: str,
    voice_id: str = "default",
    emotion: str = "neutral",
    language_id: str = "ar",
) -> np.ndarray:
    """
    Synthesize Arabic speech. Never raises — always returns audio.

    Degradation strategy:
      - Missing voice WAV  → silent fallback, base voice used (no cloning)
      - Synthesis crash    → random fallback phrase synthesized with base voice
      - Total failure      → 0.5 s of silence (numpy zeros)
    """
    try:
        model = _get_model()

        # ── Voice cloning reference ───────────────────────────────────────
        audio_prompt_path = None
        if voice_id and voice_id != "default":
            candidate = f"data/voices/{voice_id}.wav"
            if os.path.exists(candidate):
                audio_prompt_path = candidate
            else:
                _log.warning(
                    "[TTS] Voice file not found for voice_id=%s, using base voice",
                    voice_id,
                )
                # audio_prompt_path stays None — base voice, no cloning

        # ── Emotion → synthesis parameters ───────────────────────────────
        emotion_presets = {
            "neutral":    dict(exaggeration=0.5, cfg_weight=0.5),
            "friendly":   dict(exaggeration=0.6, cfg_weight=0.55),
            "happy":      dict(exaggeration=0.7, cfg_weight=0.6),
            "frustrated": dict(exaggeration=0.8, cfg_weight=0.65),
            "interested": dict(exaggeration=0.6, cfg_weight=0.6),
            "hesitant":   dict(exaggeration=0.45, cfg_weight=0.5),
        }
        preset = emotion_presets.get(emotion, emotion_presets["neutral"])

        # ── Primary synthesis attempt ─────────────────────────────────────
        try:
            wav = model.synthesize(
                text=text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                **preset,
            )
            result = wav.flatten()
            # Free temporary tensors after each synthesis to prevent VRAM fragmentation
            torch.cuda.empty_cache()
            return result

        except Exception as e:
            _is_cuda_error = "CUDA" in str(e) or "memory" in str(e).lower() or "cuda" in str(e).lower()

            if _is_cuda_error:
                # CUDA OOM or context corruption — reset singleton and clear cache
                global _tts_model, _egyptian_active
                _log.warning("[TTS] CUDA error detected, resetting model. Error: %s", e)
                _tts_model = None
                _egyptian_active = False
                torch.cuda.empty_cache()

                # Attempt one reload + retry with fallback phrase
                try:
                    fallback_text = random.choice(_FALLBACK_PHRASES)
                    fresh_model = _get_model()
                    fallback_wav = fresh_model.synthesize(
                        text=fallback_text,
                        language_id="ar",
                        audio_prompt_path=None,
                        exaggeration=0.5,
                        cfg_weight=0.5,
                    )
                    torch.cuda.empty_cache()
                    return fallback_wav.flatten()
                except Exception as reload_err:
                    _log.error("[TTS] Model reload failed: %s", reload_err)
                    sample_rate = 24000
                    return np.zeros(sample_rate // 2, dtype=np.float32)

            # Non-CUDA failure — synthesize a natural fallback phrase instead
            fallback_text = random.choice(_FALLBACK_PHRASES)
            _log.warning(
                "[TTS] Synthesis failed, returning fallback phrase. Error: %s", e
            )
            try:
                fallback_wav = model.synthesize(
                    text=fallback_text,
                    language_id="ar",
                    audio_prompt_path=None,   # base voice, no cloning
                    exaggeration=0.5,
                    cfg_weight=0.5,
                )
                torch.cuda.empty_cache()
                return fallback_wav.flatten()
            except Exception:
                sample_rate = 24000
                return np.zeros(sample_rate // 2, dtype=np.float32)

    except Exception as e:
        # Total failure (model load failed, fallback synthesis also failed, etc.)
        # Return 0.5 s of silence so the pipeline never crashes.
        _log.error("[TTS] Total failure, returning silence. Error: %s", e)
        sample_rate = 24000  # Chatterbox output sample rate
        return np.zeros(sample_rate // 2, dtype=np.float32)
