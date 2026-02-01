# scripts/verify_setup.py
"""
Verify that the VCAI environment is set up correctly.
Run this after installation to check everything works.

Usage:
    python scripts/verify_setup.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_python_version():
    """Check Python version is compatible."""
    print("\n[1/7] Checking Python version...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major != 3:
        print(f"  ❌ Python {version_str} - Need Python 3.x")
        return False
    
    if version.minor > 12:
        print(f"  ⚠️ Python {version_str} - Python 3.13+ may have compatibility issues")
        print(f"     Recommended: Python 3.10, 3.11, or 3.12")
        return False
    
    if version.minor < 10:
        print(f"  ⚠️ Python {version_str} - Python 3.10+ recommended")
        return True  # Might still work
    
    print(f"  ✅ Python {version_str}")
    return True


def check_pytorch():
    """Check PyTorch is installed with CUDA."""
    print("\n[2/7] Checking PyTorch...")
    
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ PyTorch {version}")
            print(f"  ✅ CUDA {cuda_version} available")
            print(f"  ✅ GPU: {gpu_name}")
            return True
        else:
            print(f"  ⚠️ PyTorch {version} (CPU only)")
            print(f"     For GPU support, reinstall with:")
            print(f"     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return True  # Can still work, just slower
            
    except ImportError:
        print("  ❌ PyTorch not installed")
        print("     Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_faster_whisper():
    """Check Faster-Whisper is installed."""
    print("\n[3/7] Checking Faster-Whisper...")
    
    try:
        import faster_whisper
        print(f"  ✅ Faster-Whisper installed")
        return True
    except ImportError:
        print("  ❌ Faster-Whisper not installed")
        print("     Install with: pip install faster-whisper")
        return False


def check_langgraph():
    """Check LangGraph is installed."""
    print("\n[4/7] Checking LangGraph...")
    
    try:
        import langgraph
        print(f"  ✅ LangGraph installed")
        return True
    except ImportError:
        print("  ❌ LangGraph not installed")
        print("     Install with: pip install langgraph")
        return False


def check_fastapi():
    """Check FastAPI is installed."""
    print("\n[5/7] Checking FastAPI...")
    
    try:
        import fastapi
        print(f"  ✅ FastAPI {fastapi.__version__}")
        return True
    except ImportError:
        print("  ❌ FastAPI not installed")
        print("     Install with: pip install fastapi uvicorn")
        return False


def check_project_structure():
    """Check project folders exist."""
    print("\n[6/7] Checking project structure...")
    
    required_folders = [
        "orchestration",
        "orchestration/nodes",
        "orchestration/graphs",
        "orchestration/mocks",
        "stt",
        "shared",
        "scripts"
    ]
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_exist = True
    
    for folder in required_folders:
        path = os.path.join(project_root, folder)
        if os.path.exists(path):
            print(f"  ✅ {folder}/")
        else:
            print(f"  ❌ {folder}/ - Missing!")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check main modules can be imported."""
    print("\n[7/7] Checking module imports...")
    
    modules = [
        ("shared.types", "Shared types"),
        ("shared.constants", "Shared constants"),
        ("shared.exceptions", "Shared exceptions"),
        ("orchestration.mocks", "Mocks"),
        ("orchestration.agent", "Orchestration Agent"),
        ("stt.realtime_stt", "STT module"),
    ]
    
    all_ok = True
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {description}")
        except ImportError as e:
            print(f"  ❌ {description} - {str(e)}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠️ {description} - {str(e)}")
    
    return all_ok


def main():
    """Run all checks."""
    print("="*60)
    print("VCAI Environment Verification")
    print("="*60)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("Faster-Whisper", check_faster_whisper()))
    results.append(("LangGraph", check_langgraph()))
    results.append(("FastAPI", check_fastapi()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Module Imports", check_imports()))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        if result:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1
    
    print(f"\nResult: {passed}/{len(results)} checks passed")
    
    if failed == 0:
        print("\n🎉 Environment is ready! You can now run:")
        print("   python scripts/test_orchestration.py")
        return 0
    else:
        print("\n⚠️ Some checks failed. Please fix the issues above.")
        print("   See docs/SETUP.md for detailed instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())