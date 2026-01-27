# llm/prompts.py
"""
Prompt building functions for LLM.

Builds prompts that include:
- Persona/character information
- Emotion guidance
- RAG context (property info)
- Checkpoint summaries (conversation history)
- Recent messages

IMPORTANT: Enforces Egyptian Arabic dialect (عامية مصرية)
"""

EMOTION_GUIDANCE = {
    "happy": "انت مبسوط ومتحمس للشقة",
    "angry": "انت زعلان ومش راضي خالص",
    "frustrated": "انت متنرفز من الأسعار أو الخيارات",
    "neutral": "انت هادي وبتسأل عادي",
    "fearful": "انت متردد وخايف تتنصب عليك",
    "interested": "انت مهتم وعايز تعرف أكتر",
    "hesitant": "انت مش متأكد ومحتاج تفكر"
}

# Egyptian dialect examples for the LLM to learn from
EGYPTIAN_EXAMPLES = [
    "طيب والسعر كام؟",
    "ده غالي أوي!",
    "مفيش أرخص من كده؟",
    "فين الشقة دي بالظبط؟",
    "المساحة قد إيه؟",
    "فيه تقسيط ولا لأ؟",
    "التقسيط على كام سنة؟",
    "المقدم كام؟",
    "الاستلام هيكون امتى؟",
    "طيب عايز أشوفها",
    "أنا مش متأكد",
    "خليني أفكر شوية",
    "ممكن تنزل في السعر؟",
    "كده كتير عليا",
    "المنطقة دي كويسة؟",
    "فيه مواصلات قريبة؟",
    "التشطيب إيه؟ سوبر لوكس؟",
    "أنا عايز حاجة أصغر",
    "ده مش اللي أنا بدور عليه",
    "طيب تمام، هفكر وأرد عليك",
]


def build_system_prompt(persona: dict, emotion: dict, emotional_context: dict, rag_context: dict) -> str:
    """Build system prompt for the LLM."""
    
    persona_name = persona.get("name", persona.get("name_ar", "عميل"))
    personality = persona.get("personality_prompt", "أنت عميل مصري بتدور على شقة")
    difficulty = persona.get("difficulty", "medium")
    traits = persona.get("traits", [])
    
    current_emotion = emotion.get("primary_emotion", "neutral")
    
    # Format RAG documents
    rag_docs = rag_context.get("documents", [])
    rag_info = ""
    if rag_docs:
        rag_parts = []
        for doc in rag_docs[:3]:
            content = doc.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            rag_parts.append(f"- {content}")
        rag_info = "\n".join(rag_parts)
    
    traits_str = "، ".join(traits) if traits else "عميل عادي"
    emotion_guidance = EMOTION_GUIDANCE.get(current_emotion, "انت هادي")
    
    difficulty_guidance = {
        "hard": "انت عميل صعب، بتفاصل في السعر وشكاك جداً. اسأل أسئلة صعبة.",
        "easy": "انت عميل سهل ومتعاون. بتسأل أسئلة بسيطة.",
        "medium": "انت عميل عادي، بتسأل أسئلة منطقية."
    }.get(difficulty, "انت عميل عادي")
    
    # Build examples string
    examples_str = "\n".join([f'- "{ex}"' for ex in EGYPTIAN_EXAMPLES[:10]])
    
    return f"""انت عميل مصري من القاهرة اسمك {persona_name}. بتدور على شقة تشتريها.

## مهم جداً - اللهجة:
انت لازم ترد بالعامية المصرية بتاعت القاهرة!
- قول "إيه" مش "ما" أو "ماذا"
- قول "كام" مش "كم"
- قول "فين" مش "أين"
- قول "عايز" مش "أريد"
- قول "مفيش" مش "لا يوجد"
- قول "ليه" مش "لماذا"
- قول "إزاي" مش "كيف"
- قول "ده/دي" مش "هذا/هذه"
- قول "كده" مش "هكذا"
- قول "دلوقتي" مش "الآن"

## دورك:
- انت العميل اللي عايز يشتري شقة
- البياع هو اللي بيكلمك
- رد عليه كعميل مصري

## قواعد:
1. رد بالمصري بس - ممنوع الفصحى خالص!
2. ردك يكون جملة أو اتنين بس
3. متكررش أسئلة سألتها قبل كده
4. لو البياع قالك السعر، متسألش عنه تاني

## شخصيتك:
{personality}
{emotion_guidance}
{difficulty_guidance}

## معلومات متاحة:
{rag_info if rag_info else "مفيش معلومات"}

## أمثلة لردود مصرية صح:
{examples_str}

## أمثلة غلط (متقولش كده):
- "ما هو السعر؟" ❌ (قول: "السعر كام؟")
- "أين الموقع؟" ❌ (قول: "فين المكان ده؟")
- "هل يوجد تقسيط؟" ❌ (قول: "فيه تقسيط؟")
- "أريد أن أرى الشقة" ❌ (قول: "عايز أشوفها")
- "كم المساحة؟" ❌ (قول: "المساحة قد إيه؟")

رد على البياع بالمصري:"""


def build_messages(system_prompt: str, memory: dict, salesperson_text: str) -> list:
    """
    Build message list for chat completion.
    
    IMPORTANT: Includes checkpoint summaries + recent messages
    so the LLM remembers the full conversation context.
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # ══════════════════════════════════════════════════════════════════════════
    # 1. ADD CHECKPOINT SUMMARIES (Conversation History)
    # ══════════════════════════════════════════════════════════════════════════
    checkpoints = memory.get("checkpoints", [])
    
    if checkpoints:
        checkpoint_summary_parts = []
        
        for cp in checkpoints:
            summary = cp.get("summary", "")
            key_points = cp.get("key_points", [])
            
            if summary:
                checkpoint_summary_parts.append(summary)
            
            if key_points and len(key_points) > 0:
                # Take top 3 key points
                points_str = "، ".join([str(p) for p in key_points[:3]])
                checkpoint_summary_parts.append(f"({points_str})")
        
        if checkpoint_summary_parts:
            checkpoint_context = " ".join(checkpoint_summary_parts)
            
            # Add as context about what happened before
            messages.append({
                "role": "user", 
                "content": f"[اللي فات في المحادثة: {checkpoint_context}]\n\nكمّل المحادثة:"
            })
            messages.append({
                "role": "assistant",
                "content": "تمام، فاكر كل حاجة. كمّل."
            })
    
    # ══════════════════════════════════════════════════════════════════════════
    # 2. ADD RECENT MESSAGES (Last 8 messages for immediate context)
    # ══════════════════════════════════════════════════════════════════════════
    recent_messages = memory.get("recent_messages", [])
    
    for msg in recent_messages[-8:]:
        speaker = msg.get("speaker", "")
        text = msg.get("text", "")
        
        if not text:
            continue
        
        if speaker == "salesperson":
            messages.append({"role": "user", "content": f"البياع: {text}"})
        elif speaker in ("vc", "customer"):
            messages.append({"role": "assistant", "content": text})
    
    # ══════════════════════════════════════════════════════════════════════════
    # 3. ADD CURRENT SALESPERSON MESSAGE
    # ══════════════════════════════════════════════════════════════════════════
    messages.append({"role": "user", "content": f"البياع: {salesperson_text}"})
    
    return messages


def build_evaluation_prompt(
    salesperson_text: str,
    rag_context: dict,
    conversation_history: list = None
) -> str:
    """
    Build prompt for evaluating salesperson accuracy.
    
    Uses RAG context as ground truth to check if salesperson
    gave correct information.
    """
    rag_docs = rag_context.get("documents", [])
    ground_truth = ""
    if rag_docs:
        truth_parts = []
        for doc in rag_docs[:3]:
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            truth_parts.append(f"[{source}]: {content}")
        ground_truth = "\n".join(truth_parts)
    
    return f"""انت مقيّم لأداء البائع. قارن كلام البائع بالمعلومات الصحيحة.

## كلام البائع:
"{salesperson_text}"

## المعلومات الصحيحة:
{ground_truth if ground_truth else "مفيش معلومات متاحة"}

## قيّم:
1. المعلومات صح ولا غلط؟
2. إيه الغلط لو فيه؟
3. نصيحة للبائع

رد بالمصري في جملتين:"""