# orchestration/mocks/mock_llm.py
"""
Mock LLM (Large Language Model) functions.
Replace with real imports from llm.agent when Person D completes their work.

USAGE:
    # Now (development):
    from orchestration.mocks.mock_llm import generate_response, summarize_conversation
    
    # Later (integration):
    from llm.agent import generate_response, summarize_conversation
"""

import random
from typing import Optional

from shared.types import (
    EmotionResult,
    EmotionalContext,
    Persona,
    SessionMemory,
    RAGContext,
    MemoryCheckpoint,
    Message
)


# Response templates based on different situations
RESPONSE_TEMPLATES = {
    # Greetings
    "greeting": [
        "أهلاً بيك، أنا سعيد إني أساعدك النهاردة. إيه اللي بتدور عليه؟",
        "مرحبا بيك في شركتنا، إزي أقدر أخدمك؟",
        "أهلاً وسهلاً، اتفضل احكيلي إيه اللي محتاجه",
    ],
    
    # Price questions
    "price": [
        "الأسعار بتبدأ من {price} جنيه، وممكن نتكلم في التقسيط لو حابب",
        "السعر {price} جنيه، وده سعر مناسب جداً للمنطقة دي",
        "الوحدة دي ب {price} جنيه، وفيه إمكانية خصم لو الدفع كاش",
    ],
    
    # Price objection (customer thinks it's expensive)
    "price_objection": [
        "أنا فاهم إن السعر ممكن يكون عالي، بس لو بصينا للموقع والتشطيب هتلاقي إنه مناسب جداً",
        "السعر ده فعلاً competitive للمنطقة، وكمان عندنا نظام تقسيط مريح",
        "خليني أوريك وحدات تانية ممكن تناسب ميزانيتك أكتر",
        "ممكن نتكلم مع الإدارة في موضوع السعر، إيه الميزانية اللي مرتاح فيها؟",
    ],
    
    # Location questions
    "location": [
        "الموقع ممتاز، قريب من {landmark} ومفيش زحمة كتير",
        "المنطقة دي هادية وفيها كل الخدمات، مدارس ومستشفيات وسوبر ماركت",
        "من أحسن المواقع في المنطقة، وقريب من المحاور الرئيسية",
    ],
    
    # Features questions
    "features": [
        "الشقة فيها {rooms} غرف نوم و{bathrooms} حمام، وريسيبشن واسع",
        "التشطيب سوبر لوكس، وفيه أسانسير طبعاً",
        "فيه بلكونة كبيرة وإطلالة حلوة، وموقف سيارات",
    ],
    
    # Payment/installment questions
    "payment": [
        "نظام الدفع مرن، ممكن مقدم 20% والباقي على 5 سنين",
        "لو حابب تدفع كاش فيه خصم 10%، أو تقسيط على 7 سنين",
        "المقدم يبدأ من 10% وباقي المبلغ بالتقسيط المريح",
    ],
    
    # Availability/viewing
    "viewing": [
        "الوحدة متاحة للمعاينة، إمتى يناسبك نعدي عليها؟",
        "ممكن نحدد معاد المعاينة بكرة أو بعده، إيه اللي يريحك؟",
        "خليني أأكد الموعد مع مكتب المبيعات وأرد عليك",
    ],
    
    # Hesitation handling
    "hesitation": [
        "أنا فاهم إنك محتاج وقت تفكر، خد راحتك ولو عندك أي سؤال أنا موجود",
        "الموضوع مش محتاج عجلة، بس خليني أقولك إن الوحدة عليها طلب كتير",
        "لو فيه حاجة مقلقاك قولي وأنا هحاول أساعدك",
    ],
    
    # Closing attempts
    "closing": [
        "لو موافق على الشروط دي، نقدر نحجزلك الوحدة النهاردة",
        "إيه رأيك نمشي في الإجراءات؟ الوحدة دي فرصة كويسة",
        "تمام، يعني نقدر نقول إننا اتفقنا؟",
    ],
    
    # Generic/fallback
    "generic": [
        "تمام، فيه حاجة تانية عايز تسأل عنها؟",
        "أيوه، ده كلام صح",
        "فاهم، خليني أشرحلك أكتر",
        "طيب، إيه رأيك نشوف الخيارات المتاحة؟",
    ],
    
    # Empathy responses (for frustrated customers)
    "empathy": [
        "أنا فاهم قلقك، وده طبيعي في قرار زي ده",
        "معاك حق، الموضوع محتاج تفكير، خد وقتك",
        "أنا هنا عشان أساعدك تاخد القرار الصح، مش عشان أضغط عليك",
    ],
    
    # Rushed customer responses
    "rushed": [
        "حاضر، هختصرلك: الشقة {size} متر، ب {price} جنيه، في {location}",
        "باختصار: السعر {price}، المقدم 20%، والمعاينة متاحة بكرة",
        "تمام، أهم النقط: {rooms} غرف، {price} جنيه، تقسيط 5 سنين",
    ],
}


def _detect_intent(text: str) -> str:
    """Simple intent detection based on keywords."""
    text_lower = text.lower()
    
    # Greeting
    if any(word in text_lower for word in ["مرحبا", "أهلا", "السلام", "صباح", "مساء"]):
        return "greeting"
    
    # Price
    if any(word in text_lower for word in ["سعر", "بكام", "كام", "تكلفة", "فلوس"]):
        return "price"
    
    # Price objection
    if any(word in text_lower for word in ["غالي", "كتير", "مش قادر", "ميزانية"]):
        return "price_objection"
    
    # Location
    if any(word in text_lower for word in ["فين", "موقع", "منطقة", "عنوان", "قريب"]):
        return "location"
    
    # Features
    if any(word in text_lower for word in ["غرف", "مساحة", "متر", "تشطيب", "أسانسير"]):
        return "features"
    
    # Payment
    if any(word in text_lower for word in ["تقسيط", "مقدم", "دفع", "كاش", "أقساط"]):
        return "payment"
    
    # Viewing
    if any(word in text_lower for word in ["أشوف", "معاينة", "أزور", "موعد"]):
        return "viewing"
    
    # Hesitation
    if any(word in text_lower for word in ["مش متأكد", "محتاج أفكر", "مش عارف"]):
        return "hesitation"
    
    return "generic"


def _personalize_response(response: str, rag_context: RAGContext = None) -> str:
    """Add real data from RAG context to response."""
    # Default values
    defaults = {
        "price": "850,000",
        "size": "120",
        "rooms": "3",
        "bathrooms": "2",
        "location": "التجمع الخامس",
        "landmark": "الجامعة الأمريكية"
    }
    
    # Try to extract from RAG context
    if rag_context and rag_context.get("documents"):
        doc_content = rag_context["documents"][0]["content"]
        
        # Simple extraction (in real implementation, this would be more sophisticated)
        if "850" in doc_content:
            defaults["price"] = "850,000"
        elif "1,200" in doc_content:
            defaults["price"] = "1,200,000"
        elif "550" in doc_content:
            defaults["price"] = "550,000"
        
        if "120" in doc_content:
            defaults["size"] = "120"
        elif "150" in doc_content:
            defaults["size"] = "150"
        elif "90" in doc_content:
            defaults["size"] = "90"
    
    # Replace placeholders
    for key, value in defaults.items():
        response = response.replace("{" + key + "}", value)
    
    return response


def _adjust_for_emotion(response: str, emotional_context: EmotionalContext = None) -> str:
    """Adjust response based on customer's emotional state."""
    if not emotional_context:
        return response
    
    risk_level = emotional_context.get("risk_level", "low")
    recommendation = emotional_context.get("recommendation", "")
    
    # Add empathy prefix for high-risk situations
    if risk_level == "high":
        empathy_prefix = random.choice([
            "أنا فاهم موقفك، و",
            "معاك حق، و",
            "أنا عارف إن الموضوع صعب، بس ",
        ])
        response = empathy_prefix + response[0].lower() + response[1:]
    
    return response


def generate_response(
    customer_text: str,
    emotion: EmotionResult,
    emotional_context: EmotionalContext,
    persona: Persona,
    memory: SessionMemory,
    rag_context: RAGContext
) -> str:
    """
    Mock LLM response generation.
    
    INPUT:
        customer_text: str - What the salesperson said
        emotion: EmotionResult - Detected emotion
        emotional_context: EmotionalContext - Emotional analysis
        persona: Persona - VC persona configuration
        memory: SessionMemory - Conversation memory
        rag_context: RAGContext - Retrieved documents
    
    OUTPUT:
        str - VC response in Egyptian Arabic
    """
    # Detect intent from customer text
    intent = _detect_intent(customer_text)
    
    print(f"[MOCK LLM] Detected intent: {intent}")
    print(f"[MOCK LLM] Customer emotion: {emotion.get('primary_emotion', 'unknown')}")
    print(f"[MOCK LLM] Risk level: {emotional_context.get('risk_level', 'unknown')}")
    
    # Select response template
    templates = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["generic"])
    
    # For high-risk emotional situations, prefer empathy responses
    if emotional_context.get("risk_level") == "high" and intent not in ["empathy"]:
        # Mix in some empathy
        templates = RESPONSE_TEMPLATES["empathy"] + templates
    
    # Select random response from templates
    response = random.choice(templates)
    
    # Personalize with RAG data
    response = _personalize_response(response, rag_context)
    
    # Adjust for emotion
    response = _adjust_for_emotion(response, emotional_context)
    
    print(f"[MOCK LLM] Generated response: '{response[:50]}...'")
    
    return response


def summarize_conversation(messages: list[Message]) -> str:
    """
    Mock conversation summarization for checkpoints.
    
    INPUT:
        messages: list[Message] - Messages to summarize
    
    OUTPUT:
        str - Summary in Arabic
    """
    if not messages:
        return "لا يوجد محادثة للتلخيص"
    
    # Extract key information (simple mock)
    salesperson_messages = [m for m in messages if m.get("speaker") == "salesperson"]
    
    # Build simple summary
    topics = []
    for msg in salesperson_messages:
        text = msg.get("text", "").lower()
        if any(word in text for word in ["سعر", "بكام"]):
            topics.append("سأل عن السعر")
        if any(word in text for word in ["موقع", "فين"]):
            topics.append("سأل عن الموقع")
        if any(word in text for word in ["تقسيط", "دفع"]):
            topics.append("سأل عن نظام الدفع")
        if any(word in text for word in ["غرف", "مساحة"]):
            topics.append("سأل عن المواصفات")
    
    # Remove duplicates
    topics = list(set(topics))
    
    if topics:
        summary = f"في المحادثة دي: {'. '.join(topics)}."
    else:
        summary = f"محادثة من {len(messages)} رسالة بين البائع والعميل."
    
    print(f"[MOCK LLM] Generated summary: '{summary}'")
    
    return summary


def extract_key_points(messages: list[Message]) -> list[str]:
    """
    Mock extraction of key points from conversation.
    
    INPUT:
        messages: list[Message] - Messages to analyze
    
    OUTPUT:
        list[str] - Key points mentioned
    """
    key_points = []
    
    for msg in messages:
        text = msg.get("text", "").lower()
        
        # Extract mentions of areas
        areas = ["التجمع", "زايد", "أكتوبر", "مدينة نصر", "المعادي"]
        for area in areas:
            if area in text:
                key_points.append(f"اهتمام بمنطقة {area}")
        
        # Extract price mentions
        if "غالي" in text:
            key_points.append("اعتراض على السعر")
        
        # Extract preferences
        if any(word in text for word in ["كبير", "واسع"]):
            key_points.append("يفضل المساحات الكبيرة")
        
        if "هادي" in text or "هادية" in text:
            key_points.append("يفضل المناطق الهادية")
    
    # Remove duplicates
    key_points = list(set(key_points))
    
    return key_points


# For testing
if __name__ == "__main__":
    # Test generate_response
    print("="*60)
    print("Testing LLM Response Generation")
    print("="*60)
    
    # Mock inputs
    test_inputs = [
        "مرحبا، أنا عايز أشوف شقق",
        "الشقة دي بكام؟",
        "ده غالي أوي عليا",
        "فين موقع الشقة بالظبط؟",
        "إيه نظام التقسيط؟",
        "مش متأكد، محتاج أفكر",
    ]
    
    # Mock emotion and context
    mock_emotion = {
        "primary_emotion": "neutral",
        "confidence": 0.8,
        "voice_emotion": "neutral",
        "text_emotion": "neutral",
        "intensity": "medium",
        "scores": {}
    }
    
    mock_emotional_context = {
        "current": mock_emotion,
        "trend": "stable",
        "recommendation": "be_professional",
        "risk_level": "low"
    }
    
    mock_persona = {
        "id": "test",
        "name": "عميل تجريبي",
        "personality_prompt": "test"
    }
    
    mock_memory = {
        "session_id": "test",
        "checkpoints": [],
        "recent_messages": [],
        "total_turns": 0
    }
    
    mock_rag = {
        "query": "",
        "documents": [
            {"content": "شقة 120 متر في التجمع الخامس بسعر 850,000 جنيه", "source": "test.pdf", "score": 0.9, "metadata": {}}
        ],
        "total_found": 1
    }
    
    for text in test_inputs:
        print(f"\nCustomer: '{text}'")
        response = generate_response(
            customer_text=text,
            emotion=mock_emotion,
            emotional_context=mock_emotional_context,
            persona=mock_persona,
            memory=mock_memory,
            rag_context=mock_rag
        )
        print(f"VC: '{response}'")
        print("-" * 40)
    
    # Test summarization
    print("\n" + "="*60)
    print("Testing Conversation Summarization")
    print("="*60)
    
    test_messages = [
        {"speaker": "salesperson", "text": "مرحبا، أنا عايز أشوف شقق في التجمع"},
        {"speaker": "vc", "text": "أهلاً، عندنا خيارات كتير"},
        {"speaker": "salesperson", "text": "السعر إيه تقريباً؟"},
        {"speaker": "vc", "text": "الأسعار بتبدأ من 850 ألف"},
        {"speaker": "salesperson", "text": "إيه نظام التقسيط؟"},
    ]
    
    summary = summarize_conversation(test_messages)
    print(f"Summary: {summary}")
    
    key_points = extract_key_points(test_messages)
    print(f"Key points: {key_points}")