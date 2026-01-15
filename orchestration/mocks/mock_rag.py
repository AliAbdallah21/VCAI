# orchestration/mocks/mock_rag.py
"""
Mock RAG (Retrieval-Augmented Generation) functions.
Replace with real imports from rag.agent when Person D completes their work.

USAGE:
    # Now (development):
    from orchestration.mocks.mock_rag import retrieve_context
    
    # Later (integration):
    from rag.agent import retrieve_context
"""

import random
from typing import Optional

from shared.types import RAGDocument, RAGContext


# Mock document database (simulating real estate documents)
MOCK_DOCUMENTS = [
    # Properties
    {
        "content": "شقة 120 متر في التجمع الخامس، 3 غرف نوم، 2 حمام، ريسيبشن كبير. الدور الخامس بأسانسير. تشطيب سوبر لوكس. السعر: 850,000 جنيه. إمكانية التقسيط على 5 سنوات.",
        "source": "properties_tagamo3.pdf",
        "keywords": ["تجمع", "خامس", "120", "متر", "شقة", "غرف"]
    },
    {
        "content": "شقة 150 متر في مدينة نصر، 3 غرف نوم، 2 حمام، ريسيبشن، بلكونة. الدور الثالث. تشطيب لوكس. السعر: 1,200,000 جنيه. قريبة من سيتي ستارز.",
        "source": "properties_nasr_city.pdf",
        "keywords": ["مدينة نصر", "150", "متر", "سيتي ستارز"]
    },
    {
        "content": "شقة 90 متر في المعادي، غرفتين نوم، حمام، ريسيبشن. الدور الثاني. تشطيب جيد. السعر: 650,000 جنيه. منطقة هادية وقريبة من المترو.",
        "source": "properties_maadi.pdf",
        "keywords": ["معادي", "90", "متر", "مترو"]
    },
    {
        "content": "فيلا 300 متر في الشيخ زايد، 4 غرف نوم، 3 حمام، حديقة خاصة، جراج. السعر: 4,500,000 جنيه. كمبوند راقي مع أمن 24 ساعة.",
        "source": "properties_zayed.pdf",
        "keywords": ["زايد", "فيلا", "300", "كمبوند", "حديقة"]
    },
    {
        "content": "شقة 100 متر في 6 أكتوبر، 2 غرف نوم، حمام، ريسيبشن. السعر: 550,000 جنيه. قريبة من مول مصر. تشطيب نصف تشطيب.",
        "source": "properties_october.pdf",
        "keywords": ["أكتوبر", "100", "مول مصر"]
    },
    
    # Pricing information
    {
        "content": "أسعار التقسيط: مقدم 20% والباقي على 5 سنوات بدون فوائد. مقدم 10% والباقي على 7 سنوات بفائدة 5%. إمكانية الدفع كاش بخصم 10%.",
        "source": "pricing_policy.pdf",
        "keywords": ["تقسيط", "مقدم", "فوائد", "كاش", "خصم", "سعر"]
    },
    {
        "content": "العمولة: 2.5% من قيمة الوحدة تدفع عند إتمام البيع. العمولة قابلة للتفاوض للعملاء المميزين.",
        "source": "commission_policy.pdf",
        "keywords": ["عمولة", "قيمة"]
    },
    
    # Company information
    {
        "content": "شركة الأمل للتطوير العقاري، تأسست عام 2005. أكثر من 50 مشروع ناجح. خدمة عملاء 24 ساعة. ضمان 10 سنوات على جميع الوحدات.",
        "source": "company_info.pdf",
        "keywords": ["شركة", "أمل", "ضمان", "خدمة"]
    },
    {
        "content": "مواعيد العمل: السبت للخميس من 9 صباحاً حتى 9 مساءً. الجمعة من 2 ظهراً حتى 9 مساءً. خط ساخن: 19999",
        "source": "contact_info.pdf",
        "keywords": ["مواعيد", "عمل", "خط", "ساخن"]
    },
    
    # Legal information
    {
        "content": "جميع الوحدات مسجلة في الشهر العقاري. عقود موثقة. إمكانية المعاينة قبل الشراء. ضمان استرداد المقدم خلال 14 يوم.",
        "source": "legal_info.pdf",
        "keywords": ["عقد", "شهر عقاري", "مسجل", "ضمان", "استرداد"]
    },
    
    # Neighborhoods
    {
        "content": "التجمع الخامس: منطقة راقية، قريبة من الجامعة الأمريكية. خدمات متكاملة، مولات، مدارس دولية. أسعار متوسطة إلى مرتفعة.",
        "source": "area_guide.pdf",
        "keywords": ["تجمع", "خامس", "جامعة", "أمريكية", "مولات"]
    },
    {
        "content": "الشيخ زايد: كمبوندات فاخرة، هدوء، مساحات خضراء. قريبة من مول مصر وهايبر وان. مناسبة للعائلات.",
        "source": "area_guide.pdf",
        "keywords": ["زايد", "كمبوند", "هدوء", "عائلات"]
    }
]


def _calculate_relevance(query: str, doc: dict) -> float:
    """Calculate simple relevance score based on keyword matching."""
    query_words = query.lower().split()
    keywords = doc.get("keywords", [])
    content_lower = doc["content"].lower()
    
    matches = 0
    for word in query_words:
        if word in content_lower:
            matches += 1
        for keyword in keywords:
            if word in keyword.lower():
                matches += 0.5
    
    # Normalize score
    score = min(1.0, matches / max(len(query_words), 1) * 0.5)
    
    # Add some randomness for variety
    score += random.uniform(0, 0.3)
    
    return min(0.99, score)


def retrieve_context(query: str, top_k: int = 3) -> RAGContext:
    """
    Mock RAG retrieval - returns relevant documents based on keyword matching.
    
    INPUT:
        query: str - Search query
        top_k: int - Number of documents to retrieve (1-10)
    
    OUTPUT:
        RAGContext - Retrieved documents with relevance scores
    """
    # Calculate relevance for all documents
    scored_docs = []
    for doc in MOCK_DOCUMENTS:
        score = _calculate_relevance(query, doc)
        scored_docs.append((doc, score))
    
    # Sort by score (highest first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Take top_k
    top_docs = scored_docs[:top_k]
    
    # Format as RAGDocument
    documents: list[RAGDocument] = []
    for doc, score in top_docs:
        documents.append({
            "content": doc["content"],
            "source": doc["source"],
            "score": round(score, 3),
            "metadata": {
                "keywords": doc.get("keywords", [])
            }
        })
    
    result: RAGContext = {
        "query": query,
        "documents": documents,
        "total_found": len(documents)
    }
    
    print(f"[MOCK RAG] Query: '{query}'")
    print(f"[MOCK RAG] Found {len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"[MOCK RAG]   {i+1}. {doc['source']} (score: {doc['score']:.2f})")
    
    return result


def add_document(content: str, source: str, keywords: list[str]) -> bool:
    """
    Mock function to add a document to the index.
    
    INPUT:
        content: str - Document content
        source: str - Source file name
        keywords: list[str] - Keywords for matching
    
    OUTPUT:
        bool - Success status
    """
    MOCK_DOCUMENTS.append({
        "content": content,
        "source": source,
        "keywords": keywords
    })
    print(f"[MOCK RAG] Added document: {source}")
    return True


def get_document_count() -> int:
    """Get total number of documents in the index."""
    return len(MOCK_DOCUMENTS)


# For testing
if __name__ == "__main__":
    # Test various queries
    test_queries = [
        "شقة في التجمع الخامس",
        "أسعار التقسيط",
        "فيلا في الشيخ زايد",
        "معلومات عن الشركة",
        "شقة رخيصة"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        
        result = retrieve_context(query, top_k=2)
        
        for doc in result["documents"]:
            print(f"\nSource: {doc['source']} (score: {doc['score']:.2f})")
            print(f"Content: {doc['content'][:100]}...")