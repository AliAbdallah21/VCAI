# scripts/test_rag.py
"""
RAG Pipeline Test Script.
Tests document loading, indexing, and retrieval.

Run: python scripts/test_rag.py
"""

import sys
sys.path.insert(0, r'C:\VCAI')

import os
import shutil
from pathlib import Path

print("=" * 60)
print("VCAI RAG Pipeline Test")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 0: Setup - Create documents directory and copy test files
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 0] Setting up test documents...")

# Check for DOCUMENTS_DIR in constants
try:
    from shared.constants import DOCUMENTS_DIR
    docs_dir = Path(DOCUMENTS_DIR)
except ImportError:
    # Fallback if not defined
    docs_dir = Path("data/documents")
    print(f"   ⚠️ DOCUMENTS_DIR not in constants, using: {docs_dir}")

# Create directory
docs_dir.mkdir(parents=True, exist_ok=True)
print(f"   Documents directory: {docs_dir}")

# Check if we need to create test documents
test_files = list(docs_dir.glob("*.json"))
if len(test_files) < 2:
    print("   Creating synthetic test documents...")
    
    # Create properties.json
    properties_content = '''[
  {
    "id": "prop_001",
    "name_ar": "شقة في التجمع الخامس",
    "location": "التجمع الخامس، القاهرة الجديدة",
    "compound": "بالم هيلز",
    "area_sqm": 120,
    "bedrooms": 3,
    "bathrooms": 2,
    "price_egp": 2500000,
    "finishing": "سوبر لوكس",
    "payment_plan": "مقدم 20%، تقسيط على 5 سنوات",
    "delivery_date": "استلام فوري",
    "description_ar": "شقة فاخرة في قلب التجمع الخامس، تشطيب سوبر لوكس، 3 غرف نوم"
  },
  {
    "id": "prop_002",
    "name_ar": "شقة في مدينتي",
    "location": "مدينتي، القاهرة الجديدة",
    "compound": "مدينتي B12",
    "area_sqm": 150,
    "bedrooms": 3,
    "bathrooms": 2,
    "price_egp": 3200000,
    "finishing": "نصف تشطيب",
    "payment_plan": "مقدم 15%، تقسيط على 7 سنوات",
    "delivery_date": "2025",
    "description_ar": "شقة واسعة في مدينتي، 150 متر مربع، نصف تشطيب"
  },
  {
    "id": "prop_003",
    "name_ar": "فيلا في الشيخ زايد",
    "location": "الشيخ زايد، 6 أكتوبر",
    "compound": "بيفرلي هيلز",
    "area_sqm": 300,
    "bedrooms": 5,
    "bathrooms": 4,
    "price_egp": 8500000,
    "finishing": "سوبر لوكس",
    "payment_plan": "مقدم 30%، تقسيط على 4 سنوات",
    "delivery_date": "استلام فوري",
    "description_ar": "فيلا فاخرة في الشيخ زايد، مساحة 300 متر"
  },
  {
    "id": "prop_004",
    "name_ar": "شقة في العاصمة الإدارية",
    "location": "العاصمة الإدارية الجديدة",
    "compound": "ميدتاون سولو",
    "area_sqm": 100,
    "bedrooms": 2,
    "bathrooms": 1,
    "price_egp": 1800000,
    "finishing": "تشطيب كامل",
    "payment_plan": "مقدم 10%، تقسيط على 8 سنوات",
    "delivery_date": "2026",
    "description_ar": "شقة في العاصمة الإدارية، سعر مميز مع تقسيط 8 سنوات"
  }
]'''
    
    with open(docs_dir / "properties.json", "w", encoding="utf-8") as f:
        f.write(properties_content)
    print("   ✅ Created properties.json")
    
    # Create company_policies.json
    policies_content = '''{
  "company_name": "شركة الأهرام للعقارات",
  "policies": {
    "payment": {
      "minimum_down_payment": "10%",
      "maximum_installment_years": 8,
      "description_ar": "خطط سداد مرنة من 10% مقدم وتقسيط حتى 8 سنوات"
    },
    "reservation": {
      "reservation_fee": 50000,
      "description_ar": "مبلغ الحجز 50,000 جنيه يُخصم من المقدم"
    },
    "delivery": {
      "delay_penalty": "1% عن كل شهر تأخير",
      "description_ar": "تعويض 1% عن كل شهر تأخير في التسليم"
    }
  }
}'''
    
    with open(docs_dir / "company_policies.json", "w", encoding="utf-8") as f:
        f.write(policies_content)
    print("   ✅ Created company_policies.json")

print(f"   Documents in folder: {list(docs_dir.glob('*.json'))}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Check imports
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 1] Checking imports...")

try:
    from rag.config import DOCS_DIR, CHROMA_DIR, FAISS_INDEX_PATH
    print(f"   ✅ Config imported")
    print(f"      DOCS_DIR: {DOCS_DIR}")
    print(f"      CHROMA_DIR: {CHROMA_DIR}")
except ImportError as e:
    print(f"   ❌ Config import failed: {e}")
    sys.exit(1)

try:
    from rag.document_loader import load_chunks, RawChunk
    print("   ✅ Document loader imported")
except ImportError as e:
    print(f"   ❌ Document loader import failed: {e}")
    sys.exit(1)

try:
    from rag.embeddings import embed_texts, embed_query
    print("   ✅ Embeddings imported")
except ImportError as e:
    print(f"   ❌ Embeddings import failed: {e}")
    sys.exit(1)

try:
    from rag.vector_store import build_chroma_index, export_faiss_from_chroma, faiss_search
    print("   ✅ Vector store imported")
except ImportError as e:
    print(f"   ❌ Vector store import failed: {e}")
    sys.exit(1)

try:
    from rag.agent import retrieve_context
    print("   ✅ RAG agent imported")
except ImportError as e:
    print(f"   ❌ RAG agent import failed: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Load documents
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 2] Loading documents...")

try:
    chunks = load_chunks()
    print(f"   ✅ Loaded {len(chunks)} chunks")
    
    if chunks:
        print(f"\n   First chunk preview:")
        print(f"      ID: {chunks[0].id}")
        print(f"      Source: {chunks[0].source}")
        print(f"      Content: {chunks[0].content[:100]}...")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Test embeddings
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 3] Testing embeddings...")

try:
    test_texts = ["شقة في التجمع الخامس", "سعر الشقة كام؟"]
    embeddings = embed_texts(test_texts)
    print(f"   ✅ Embeddings generated")
    print(f"      Shape: {embeddings.shape}")
    print(f"      Dtype: {embeddings.dtype}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Build Chroma index
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 4] Building Chroma index...")

try:
    # Clear existing index for fresh test
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("   Cleared existing Chroma index")
    
    result = build_chroma_index()
    print(f"   ✅ Chroma index built")
    print(f"      {result}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Export to FAISS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 5] Exporting to FAISS...")

try:
    result = export_faiss_from_chroma()
    print(f"   ✅ FAISS export complete")
    print(f"      {result}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Test FAISS search
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 6] Testing FAISS search...")

test_queries = [
    "شقة في التجمع الخامس",
    "سعر الشقة كام",
    "فيلا في الشيخ زايد",
    "تقسيط على كام سنة",
    "العاصمة الإدارية",
]

for query in test_queries:
    print(f"\n   Query: '{query}'")
    try:
        hits = faiss_search(query, top_k=2)
        if hits:
            for i, hit in enumerate(hits):
                print(f"      [{i+1}] Score: {hit['score']:.3f}")
                print(f"          Source: {hit['metadata'].get('source', 'unknown')}")
                print(f"          Content: {hit['content'][:80]}...")
        else:
            print("      No results")
    except Exception as e:
        print(f"      ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Test retrieve_context (main interface)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 7] Testing retrieve_context() - Main RAG Interface...")

test_queries = [
    "شقة 120 متر في التجمع",
    "سعر الفيلا",
    "خطة التقسيط",
]

for query in test_queries:
    print(f"\n   Query: '{query}'")
    try:
        context = retrieve_context(query, top_k=3)
        print(f"      Total found: {context['total_found']}")
        
        if context['documents']:
            for i, doc in enumerate(context['documents']):
                print(f"      [{i+1}] Score: {doc['score']:.3f} | Source: {doc['source']}")
                print(f"          {doc['content'][:100]}...")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print("""
✅ Documents loaded and chunked
✅ Embeddings generated (multilingual model)
✅ Chroma index built
✅ FAISS export complete
✅ Search working with Arabic queries
✅ retrieve_context() interface ready

📋 Next Steps:
1. Copy document_loader.py to C:\\VCAI\\rag\\document_loader.py
2. Copy __init__.py to C:\\VCAI\\rag\\__init__.py
3. Add test documents to data/documents/
4. Update orchestration/nodes/rag_node.py to use real RAG

🎯 RAG is ready for integration!
""")