# llm/prompts.py
"""Prompt building functions for LLM"""


EMOTION_GUIDANCE = {
    "happy": "انت مبسوط ومتحمس للشقة",
    "angry": "انت زعلان ومش راضي",
    "frustrated": "انت محبط من الأسعار أو الخيارات",
    "neutral": "انت هادي وبتسأل عادي",
    "fearful": "انت متردد وخايف تتنصب عليك",
    "interested": "انت مهتم وعايز تعرف أكتر",
    "hesitant": "انت مش متأكد ومحتاج تفكر"
}


def build_system_prompt(persona: dict, emotion: dict, emotional_context: dict, rag_context: dict) -> str:
    """Build system prompt for the LLM."""
    
    persona_name = persona.get("name", "عميل")
    personality = persona.get("personality_prompt", "أنت عميل في شركة عقارات")
    difficulty = persona.get("difficulty", "medium")
    traits = persona.get("traits", [])
    
    current_emotion = emotion.get("primary_emotion", "neutral")
    
    rag_docs = rag_context.get("documents", [])
    rag_info = "\n".join([f"- {doc.get('content', '')}" for doc in rag_docs[:3]])
    
    traits_str = "، ".join(traits) if traits else "عميل عادي"
    emotion_guidance = EMOTION_GUIDANCE.get(current_emotion, "انت هادي")
    
    difficulty_guidance = {
        "hard": "انت عميل صعب، فاصل في السعر وشكاك",
        "easy": "انت عميل سهل، مهتم ومتعاون",
        "medium": "انت عميل عادي"
    }.get(difficulty, "انت عميل عادي")
    
    return f"""أنت عميل مصري اسمك {persona_name}. انت بتدور على شقة تشتريها.

## دورك:
- انت العميل اللي عايز يشتري شقة
- البياع هو اللي بيكلمك وبيعرض عليك شقق
- انت بترد على البياع كعميل

## قواعد صارمة جداً:
- رد بالعامية المصرية فقط
- ممنوع أي لغة غير العربية (لا انجليزي ولا عبري ولا أي حاجة تانية)
- ردك يكون جملة أو جملتين بس
- لا تعرض مساعدة على البياع - انت اللي محتاج مساعدة
- اسأل عن السعر والمكان والمساحة
- لا تقول "هساعدك" أو "هعرفك" - انت العميل مش البياع

## شخصيتك:
{personality}
{emotion_guidance}
{difficulty_guidance}

## معلومات عن الشقق المتاحة:
{rag_info if rag_info else "مفيش معلومات"}

## أمثلة لردود صحيحة:
- "طيب والسعر كام؟"
- "المنطقة دي آمنة؟"
- "فيه تقسيط؟"
- "غالي أوي، مفيش أرخص؟"
- "تمام، عايز أشوفها"

رد على البياع كعميل:"""


def build_messages(system_prompt: str, memory: dict, salesperson_text: str) -> list:
    """Build message list for chat completion."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    recent_messages = memory.get("recent_messages", [])
    for msg in recent_messages[-4:]:
        speaker = msg.get("speaker", "")
        text = msg.get("text", "")
        
        if speaker == "salesperson":
            # Salesperson = user input to the AI
            messages.append({"role": "user", "content": f"البياع: {text}"})
        elif speaker == "vc" or speaker == "customer":
            # Customer (AI) = assistant response
            messages.append({"role": "assistant", "content": text})
    
    # Current salesperson message
    messages.append({"role": "user", "content": f"البياع: {salesperson_text}"})
    return messages