# scripts/validate_all.py
"""
Master validation script - Run this to test ALL implementations.
Use this after teammates push their code.

Usage:
    cd C:\VCAI
    python scripts/validate_all.py
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_SCRIPTS = [
    ("TTS (Person B)", "scripts/teammate_tests/test_tts.py"),
    ("Emotion (Person C)", "scripts/teammate_tests/test_emotion.py"),
    ("RAG (Person D)", "scripts/teammate_tests/test_rag.py"),
    ("Memory (Person D)", "scripts/teammate_tests/test_memory.py"),
    ("LLM (Person D)", "scripts/teammate_tests/test_llm.py"),
]


def run_test(name, script_path):
    """Run a test script and return pass/fail"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check if all tests passed
        if "ALL TESTS PASSED" in result.stdout:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return False


def main():
    print("=" * 60)
    print("VCAI - Full Validation Suite")
    print("=" * 60)
    
    results = []
    
    for name, script in TEST_SCRIPTS:
        if os.path.exists(script):
            passed = run_test(name, script)
            results.append((name, passed))
        else:
            print(f"\n⚠️ Script not found: {script}")
            results.append((name, None))
    
    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        if passed is True:
            status = "✅ PASSED"
        elif passed is False:
            status = "❌ FAILED"
        else:
            status = "⚠️ SKIPPED"
        print(f"  {status} - {name}")
    
    passed_count = sum(1 for _, p in results if p is True)
    failed_count = sum(1 for _, p in results if p is False)
    skipped_count = sum(1 for _, p in results if p is None)
    
    print(f"\nTotal: {passed_count} passed, {failed_count} failed, {skipped_count} skipped")
    
    if failed_count == 0 and skipped_count == 0:
        print("\n🎉 ALL COMPONENTS READY FOR INTEGRATION!")
    elif failed_count > 0:
        print("\n⚠️ Some components need fixes before integration.")


if __name__ == "__main__":
    main()
