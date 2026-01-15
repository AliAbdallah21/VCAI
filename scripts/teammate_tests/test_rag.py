# scripts/teammate_tests/test_rag.py
"""
Test script for Person D: RAG (Retrieval Augmented Generation)
Run this to validate your implementation before pushing.

Usage:
    cd C:\VCAI
    python scripts/teammate_tests/test_rag.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_CONTEXT_KEYS = ["query", "documents", "total_found"]
REQUIRED_DOC_KEYS = ["content", "source", "score", "metadata"]

TEST_QUERIES = [
    "شقق في التجمع الخامس",
    "أسعار الشقق",
    "مساحات متاحة",
]


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_function_exists():
    """Test 1: Check if retrieve_context function exists"""
    print("\n[Test 1] Checking if retrieve_context function exists...")
    
    try:
        from rag.rag_agent import retrieve_context
        print("   ✅ Function imported successfully")
        return True, retrieve_context
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   Make sure you created: rag/rag_agent.py")
        print("   With function: def retrieve_context(query: str, top_k: int = 3) -> dict")
        return False, None


def test_function_signature(func):
    """Test 2: Check function signature"""
    print("\n[Test 2] Checking function signature...")
    
    import inspect
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    if 'query' in params:
        print(f"   ✅ Parameters: {params}")
        return True
    else:
        print(f"   ❌ Missing 'query' parameter")
        return False


def test_basic_call(func):
    """Test 3: Basic function call"""
    print("\n[Test 3] Testing basic function call...")
    
    try:
        result = func("شقق في مدينة نصر")
        print("   ✅ Function returned without error")
        return True, result
    except Exception as e:
        print(f"   ❌ Function raised exception: {e}")
        return False, None


def test_return_type(result):
    """Test 4: Check return type"""
    print("\n[Test 4] Checking return type...")
    
    if isinstance(result, dict):
        print("   ✅ Return type is dict")
        return True
    else:
        print(f"   ❌ Expected dict, got {type(result)}")
        return False


def test_required_keys(result):
    """Test 5: Check required keys"""
    print("\n[Test 5] Checking required keys...")
    
    missing = [k for k in REQUIRED_CONTEXT_KEYS if k not in result]
    
    if not missing:
        print(f"   ✅ All keys present: {REQUIRED_CONTEXT_KEYS}")
        return True
    else:
        print(f"   ❌ Missing keys: {missing}")
        return False


def test_documents_list(result):
    """Test 6: Check documents is a list"""
    print("\n[Test 6] Checking documents field...")
    
    docs = result.get("documents")
    
    if not isinstance(docs, list):
        print(f"   ❌ documents should be list, got {type(docs)}")
        return False
    
    print(f"   ✅ documents is list with {len(docs)} items")
    return True


def test_document_structure(result):
    """Test 7: Check document structure"""
    print("\n[Test 7] Checking document structure...")
    
    docs = result.get("documents", [])
    
    if not docs:
        print("   ⚠️ No documents returned (might be OK if no data)")
        return True
    
    for i, doc in enumerate(docs):
        missing = [k for k in REQUIRED_DOC_KEYS if k not in doc]
        if missing:
            print(f"   ❌ Document {i} missing keys: {missing}")
            return False
        
        # Check score is float 0-1
        score = doc.get("score")
        if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            print(f"   ❌ Document {i} invalid score: {score}")
            return False
    
    print(f"   ✅ All {len(docs)} documents have correct structure")
    return True


def test_top_k_parameter(func):
    """Test 8: Test top_k parameter"""
    print("\n[Test 8] Testing top_k parameter...")
    
    try:
        result = func("شقق", top_k=5)
        docs = result.get("documents", [])
        
        if len(docs) <= 5:
            print(f"   ✅ top_k=5 returned {len(docs)} documents")
            return True
        else:
            print(f"   ❌ top_k=5 but got {len(docs)} documents")
            return False
    except Exception as e:
        print(f"   ❌ Exception with top_k: {e}")
        return False


def test_different_queries(func):
    """Test 9: Test with different queries"""
    print("\n[Test 9] Testing different queries...")
    
    all_passed = True
    for query in TEST_QUERIES:
        try:
            result = func(query)
            if isinstance(result, dict) and "documents" in result:
                print(f"   ✅ '{query}' -> {len(result['documents'])} docs")
            else:
                print(f"   ❌ Invalid result for '{query}'")
                all_passed = False
        except Exception as e:
            print(f"   ❌ Exception for '{query}': {e}")
            all_passed = False
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("RAG (Retrieval) Validation Tests")
    print("Person D Implementation")
    print("=" * 60)
    
    results = []
    
    passed, func = test_function_exists()
    results.append(("Function exists", passed))
    if not passed:
        return
    
    results.append(("Function signature", test_function_signature(func)))
    
    passed, result = test_basic_call(func)
    results.append(("Basic call", passed))
    if not passed:
        return
    
    results.append(("Return type", test_return_type(result)))
    results.append(("Required keys", test_required_keys(result)))
    results.append(("Documents list", test_documents_list(result)))
    results.append(("Document structure", test_document_structure(result)))
    results.append(("top_k parameter", test_top_k_parameter(func)))
    results.append(("Different queries", test_different_queries(func)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")


if __name__ == "__main__":
    main()
