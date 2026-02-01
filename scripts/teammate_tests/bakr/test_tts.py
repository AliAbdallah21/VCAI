# C:\VCAI\scripts\teammate_tests\bakr\test_tts.py
"""
Test script for TTS module validation
Tests against shared/interfaces.py requirements
"""

import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[0]}")

# Output directory for test audio
OUTPUT_DIR = Path(__file__).parent / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_imports():
    """Test 1: Can we import the TTS modules?"""
    print("\n" + "="*60)
    print("TEST 1: Import TTS modules")
    print("="*60)
    
    try:
        from tts.agent import text_to_speech
        print("✅ Imported: tts.agent.text_to_speech")
        
        from tts.chatterbox_model import ChatterboxTTSModel, ChatterboxTTSConfig
        print("✅ Imported: tts.chatterbox_model.ChatterboxTTSModel")
        print("✅ Imported: tts.chatterbox_model.ChatterboxTTSConfig")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_interface_signature():
    """Test 2: Does text_to_speech match interfaces.py signature?"""
    print("\n" + "="*60)
    print("TEST 2: Check function signature vs interfaces.py")
    print("="*60)
    
    from tts.agent import text_to_speech
    import inspect
    
    sig = inspect.signature(text_to_speech)
    params = list(sig.parameters.keys())
    
    print(f"Current signature: text_to_speech({', '.join(params)})")
    
    # Expected from interfaces.py:
    # def text_to_speech(text: str, voice_id: str = "default", emotion: str = "neutral", language_id: str = "ar") -> np.ndarray:
    
    expected_params = ["text", "voice_id", "emotion", "language_id"]
    
    missing = [p for p in expected_params if p not in params]
    extra = [p for p in params if p not in expected_params]
    
    if missing:
        print(f"⚠️ Missing parameters: {missing}")
    else:
        print("✅ All required parameters present")
    
    if extra:
        print(f"ℹ️ Extra parameters (OK): {extra}")
    
    # Check if language_id is missing
    if "language_id" not in params:
        print("\n⚠️ ISSUE: 'language_id' parameter missing from text_to_speech()")
        print("   interfaces.py expects: language_id: str = 'ar'")
        print("   Current code hardcodes language_id='ar' inside the function")
        return False
    
    return len(missing) == 0


def test_model_loading():
    """Test 3: Can we load the TTS model?"""
    print("\n" + "="*60)
    print("TEST 3: Load TTS model")
    print("="*60)
    
    try:
        from tts.chatterbox_model import ChatterboxTTSModel, ChatterboxTTSConfig
        
        print("Creating config...")
        cfg = ChatterboxTTSConfig(
            device="cuda",
            default_language_id="ar",
        )
        print(f"  device: {cfg.device}")
        print(f"  default_language_id: {cfg.default_language_id}")
        print(f"  sample_rate_out: {cfg.sample_rate_out}")
        
        print("\nLoading model (this may take a moment)...")
        model = ChatterboxTTSModel(cfg)
        model.load()
        
        print(f"✅ Model loaded successfully!")
        print(f"  Model sample rate: {model.sr}")
        
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_synthesis(model=None):
    """Test 4: Can we synthesize Arabic speech?"""
    print("\n" + "="*60)
    print("TEST 4: Synthesize Arabic speech")
    print("="*60)
    
    if model is None:
        from tts.chatterbox_model import ChatterboxTTSModel, ChatterboxTTSConfig
        cfg = ChatterboxTTSConfig(device="cuda", default_language_id="ar")
        model = ChatterboxTTSModel(cfg)
        model.load()
    
    test_texts = [
        ("مرحبا كيف حالك", "hello"),
        ("الشقة دي في موقع ممتاز", "apartment"),
        ("انا بحب مصر جدا", "egypt"),
    ]
    
    results = []
    
    for text, name in test_texts:
        print(f"\nGenerating: '{text}'")
        try:
            wav = model.synthesize(
                text=text,
                language_id="ar",
                exaggeration=0.5,
                cfg_weight=0.5,
            )
            
            # Check output
            print(f"  Type: {type(wav)}")
            print(f"  Shape: {wav.shape}")
            print(f"  Dtype: {wav.dtype}")
            print(f"  Duration: {len(wav.flatten()) / model.sr:.2f}s")
            
            # Validate output format
            if not isinstance(wav, np.ndarray):
                print(f"  ⚠️ Output should be np.ndarray, got {type(wav)}")
            
            if wav.dtype != np.float32:
                print(f"  ⚠️ Output should be float32, got {wav.dtype}")
            
            # Save audio
            output_path = OUTPUT_DIR / f"test_{name}.wav"
            wav_1d = wav.flatten()
            sf.write(output_path, wav_1d, model.sr)
            print(f"  ✅ Saved: {output_path}")
            
            results.append(True)
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    return all(results)


def test_agent_interface():
    """Test 5: Test the agent interface (text_to_speech function)"""
    print("\n" + "="*60)
    print("TEST 5: Test text_to_speech() agent interface")
    print("="*60)
    
    try:
        from tts.agent import text_to_speech
        
        test_cases = [
            {"text": "مرحبا بيك", "voice_id": "default", "emotion": "neutral"},
            {"text": "ده سعر كويس جدا", "voice_id": "default", "emotion": "happy"},
            {"text": "مش متأكد من الكلام ده", "voice_id": "default", "emotion": "hesitant"},
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\nTest case {i+1}: {case}")
            
            wav = text_to_speech(**case)
            
            print(f"  Type: {type(wav)}")
            print(f"  Shape: {wav.shape}")
            print(f"  Dtype: {wav.dtype}")
            
            # Check if 1D
            if wav.ndim != 1:
                print(f"  ⚠️ Output should be 1D, got {wav.ndim}D")
                wav = wav.flatten()
            
            # Save
            output_path = OUTPUT_DIR / f"test_agent_{i+1}_{case['emotion']}.wav"
            sf.write(output_path, wav, 24000)  # Assuming 24kHz
            print(f"  ✅ Saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_format():
    """Test 6: Validate output matches interfaces.py spec"""
    print("\n" + "="*60)
    print("TEST 6: Validate output format vs interfaces.py")
    print("="*60)
    
    print("""
    interfaces.py specifies:
    OUTPUT:
        np.ndarray:
            - Shape: (n_samples,) - 1D array
            - Sample rate: 24000 Hz
            - Dtype: float32
    """)
    
    try:
        from tts.agent import text_to_speech
        
        wav = text_to_speech("تجربة", voice_id="default", emotion="neutral")
        
        checks = {
            "Is np.ndarray": isinstance(wav, np.ndarray),
            "Is 1D": wav.ndim == 1,
            "Is float32": wav.dtype == np.float32,
            "Has samples": len(wav) > 0,
        }
        
        print("\nOutput validation:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}: {passed}")
        
        if not checks["Is 1D"]:
            print(f"\n  ⚠️ Shape is {wav.shape}, should be (n_samples,)")
            print(f"  Fix: Add .flatten() or .squeeze() to output")
        
        return all(checks.values())
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def generate_fix_suggestions():
    """Generate code fix suggestions"""
    print("\n" + "="*60)
    print("SUGGESTED FIXES for tts/agent.py")
    print("="*60)
    
    fix = '''
# Add language_id parameter to match interfaces.py:

def text_to_speech(
    text: str, 
    voice_id: str = "default", 
    emotion: str = "neutral",
    language_id: str = "ar"  # ADD THIS
) -> np.ndarray:
    """..."""
    model = _get_model()
    
    # ... voice_id mapping ...
    
    # ... emotion presets ...
    
    wav = model.synthesize(
        text=text,
        language_id=language_id,  # USE PARAMETER instead of hardcoded "ar"
        audio_prompt_path=audio_prompt_path,
        **preset,
    )
    
    # Ensure 1D output
    return wav.flatten()  # ADD THIS to ensure 1D
'''
    print(fix)


def main():
    print("="*60)
    print("VCAI TTS Module Validation")
    print("="*60)
    
    results = {}
    
    # Test 1: Imports
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n❌ Cannot proceed without imports")
        return
    
    # Test 2: Signature
    results["signature"] = test_interface_signature()
    
    # Test 3: Model loading
    model = test_model_loading()
    results["model_loading"] = model is not None
    
    if model:
        # Test 4: Synthesis
        results["synthesis"] = test_synthesis(model)
        
        # Test 5: Agent interface
        results["agent_interface"] = test_agent_interface()
        
        # Test 6: Output format
        results["output_format"] = test_output_format()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        generate_fix_suggestions()
    
    print(f"\n🎧 Test audio files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()