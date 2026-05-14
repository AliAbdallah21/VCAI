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
    "happy": "البياع بيتكلم بطريقة مبسوطة ومريحة — رد بإيجابية واسأل سؤال مهتم.",
    "angry": "البياع بيتكلم بطريقة متوترة — رد بهدوء لكن اسأل سؤال محدد عن اللي قاله.",
    "frustrated": "البياع بيبان متضايق — رد باهتمام واسأل سؤال يوضح حاجة مش واضحة.",
    "neutral": "البياع بيتكلم عادي — اسأل سؤال منطقي عن اللي قاله.",
    "fearful": "البياع بيتكلم بتردد — خليه يحس إنك محتاج تفاصيل أكتر قبل ما تقرر.",
    "interested": "البياع بيبان مهتم — رد باهتمام متبادل واسأل تفاصيل عن العرض.",
    "hesitant": "البياع مش واثق — اسأله سؤال مباشر يخليه يوضح العرض أكتر.",
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


def _extract_already_mentioned(memory: dict) -> list:
    """
    Extract topics/information already mentioned in the conversation.
    Looks at BOTH recent messages and checkpoint summaries so long
    conversations don't lose track of what was discussed. Also scans
    assistant (VC) messages so the AI doesn't repeat its own questions.
    """
    # Topic → keywords. Match if ANY keyword present in the text.
    TOPIC_KEYWORDS = {
        "السعر":             ["سعر", "مليون", "ألف", "جنيه", "تمن", "بكام", "كام", "غالي", "رخيص"],
        "المساحة":           ["متر", "مساحة", "قد إيه", "حجم"],
        "المكان":            ["فين", "مكان", "منطقة", "عنوان", "مدينة نصر", "التجمع", "الشيخ زايد", "القاهرة الجديدة"],
        "الدور/الطابق":      ["دور", "طابق", "أرضي"],
        "المقدم/التقسيط":    ["مقدم", "تقسيط", "قسط", "كاش", "دفع", "فايدة", "فوايد"],
        "الأسانسير":         ["أسانسير", "اسانسير", "أسنسير", "asansir", "elevator", "مصعد"],
        "الحمامات":          ["حمام", "حمامات", "تواليت"],
        "الغرف":             ["غرف", "غرفة", "أوضة", "أوض", "غراف", "نوم"],
        "المطبخ":            ["مطبخ", "مطابخ"],
        "البلكونة/الشرفة":   ["بلكونة", "شرفة", "تراس", "بلكون"],
        "التكييف":           ["تكييف", "تكيف", "مكيف", "كنديشن"],
        "التسليم/الأوراق":   ["تسليم", "استلام", "عقد", "عقود", "أوراق", "ملكية", "مالك", "تسجيل"],
        "الصيانة":           ["صيانة", "اشتراك شهري", "خدمة شهرية"],
        "المواصلات":         ["مواصلات", "اتوبيس", "مترو", "مدارس", "مدرسة", "مستشفى"],
    }

    def _scan(text: str, bag: set) -> None:
        if not text:
            return
        t = text.lower()
        for topic, kws in TOPIC_KEYWORDS.items():
            if any(kw in t for kw in kws):
                bag.add(topic)

    mentioned: set = set()

    # 1. Recent messages — both speakers count (we don't want the AI to
    #    re-ask about something it already asked itself either).
    for msg in memory.get("recent_messages", []) or []:
        _scan(msg.get("text", ""), mentioned)

    # 2. Checkpoint summaries + key points — long conversations rely on
    #    these once raw messages roll out of the recent window.
    for cp in memory.get("checkpoints", []) or []:
        _scan(cp.get("summary", ""), mentioned)
        for kp in cp.get("key_points", []) or []:
            _scan(str(kp), mentioned)

    return sorted(mentioned)


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
    
    return f"""أنت عميل مصري اسمك {persona_name} بتدور على شقة تشتريها. أنت العميل مش البياع.

شخصيتك: {personality}
حالتك: {emotion_guidance}
{difficulty_guidance}

اللهجة: عامية مصرية قاهرية فقط. قول "إيه/كام/فين/عايز/مفيش/ده/كده" مش الفصحى.

معلومات العقارات: {rag_info if rag_info else "اسأل البياع عن التفاصيل"}

أمثلة ردود صح:
- "مليون ونص ده غالي شوية. فيه حاجة أرخص في نفس المنطقة؟"
- "التجمع كويسة! المواصلات إيه هناك؟"
- "وعليكم السلام. أنا بدور على شقة 3 غرف، عندك إيه؟"
- "120 متر حلو. الدور إيه؟"

ردودك لازم تكون متعلقة بالمحادثة، مش ردود عامة."""


# Seller phrases that signal "let's close" — when any of these appears in the
# salesperson's latest message, the customer should stop asking and decide.
_CLOSURE_CUES = [
    "نمضي العقد", "نمضى العقد", "نمضو العقد", "نمضو العقود",
    "تشرفنا في المكتب", "تشرفنا بالمكتب", "تيجي المكتب",
    "تيجي تشوف", "تعالى المكتب", "تعالا المكتب",
    "نتفق", "نتقابل", "ميعاد العقد",
    "هنوقع", "هنمضي", "نوقع العقد",
]


def _seller_is_closing(salesperson_text: str) -> bool:
    if not salesperson_text:
        return False
    t = salesperson_text.lower()
    return any(cue in t for cue in _CLOSURE_CUES)


def _build_turn_instruction(topic_count: int, seller_closing: bool) -> str:
    """
    Build the per-turn instruction. Switches mode based on how much
    of the deal has already been covered — early conversation defaults
    to questions; once enough is on the table the model is pushed to
    a decision instead.
    """
    if seller_closing:
        # Seller has explicitly invited closure — strongest signal to decide.
        return (
            "[البياع بيعرض عليك تقفل الصفقة دلوقتي. لازم تختار:\n"
            "1. لو راضي بالعرض حسب معايير شخصيتك (متى توافق): قول 'تمام، أنا موافق' "
            "أو 'يلا نمشي للعقد'.\n"
            "2. لو في حاجة مش مناسبة (متى ترفض): قول 'شكراً، هفكر تاني' أو "
            "'العرض ده مش هياخده'.\n"
            "ممنوع تسأل سؤال جديد. لازم تقرر دلوقتي. الرد جملة لجملتين بحد أقصى.]"
        )

    if topic_count >= 5:
        # Plenty already discussed — bias toward decision, allow at most one
        # last question if absolutely needed.
        return (
            f"[في {topic_count} موضوع اتغطّى من المحادثة. الحالة دلوقتي:\n"
            "اختار واحد:\n"
            "1. لو المعلومات كافية حسب شخصيتك (راجع 'متى توافق'): اقفل الصفقة. "
            "قول 'تمام، أنا موافق' أو 'يلا نتفق'.\n"
            "2. لو لقيت حاجة فوق حدودك (راجع 'متى ترفض'): اعتذر بأدب وامشي.\n"
            "3. لو فعلاً ضروري سؤال أخير واحد بس عن حاجة لسه ما اتكلمتش عنها: اسأل، "
            "بس بعدها لازم تقرر في الرد اللي بعده.\n"
            "ممنوع تكرر نفس النوع من الأسئلة. الرد جملة لجملتين بحد أقصى.]"
        )

    # Early conversation — gather information naturally.
    return (
        "[رد طبيعي حسب شخصيتك وحالة المحادثة:\n"
        "• علّق على اللي قاله البياع، واسأل سؤال جديد عن حاجة لسه ما اتكلمتش عنها.\n"
        "• لو لقيت إن العرض كله عاجبك حسب 'متى توافق': اقفل الصفقة دلوقتي.\n"
        "• لو في حاجة فوق ميزانيتك أو غير واضحة: اعتذر بأدب وامشي.\n"
        "الرد جملة لجملتين بحد أقصى. ممنوع ترد بكلمة واحدة لوحدها.]"
    )


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
                "content": f"[ملخص اللي فات: {checkpoint_context}]"
            })
            messages.append({
                "role": "assistant",
                "content": "تمام، فاكر."
            })
    
    # ══════════════════════════════════════════════════════════════════════════
    # 2. ADD RECENT MESSAGES (Last 10 messages for immediate context)
    # ══════════════════════════════════════════════════════════════════════════
    recent_messages = memory.get("recent_messages", [])
    
    # Take last 10 instead of 8 for better context
    for msg in recent_messages[-10:]:
        speaker = msg.get("speaker", "")
        text = msg.get("text", "")
        
        if not text:
            continue
        
        if speaker == "salesperson":
            messages.append({"role": "user", "content": f"البياع: {text}"})
        elif speaker in ("vc", "customer"):
            messages.append({"role": "assistant", "content": text})
    
    # ══════════════════════════════════════════════════════════════════════════
    # 3. CONTEXT-AWARE PER-TURN INSTRUCTION
    # The instruction adapts to conversation state — early turns ask questions,
    # later turns push for a decision. Also detects when the seller has invited
    # closure and forces an immediate decide-or-walk-away response.
    # Putting this at the decision point (not just in the system prompt) is
    # essential — with streaming, the model commits to a path before fully
    # consuming a long system prompt.
    # ══════════════════════════════════════════════════════════════════════════
    already_mentioned = _extract_already_mentioned(memory)
    seller_closing = _seller_is_closing(salesperson_text)
    turn_instruction = _build_turn_instruction(len(already_mentioned), seller_closing)

    repeat_warning = ""
    if already_mentioned:
        mentioned_str = "، ".join(already_mentioned)
        repeat_warning = (
            f"[المواضيع اللي اتغطّت قبل كده: {mentioned_str}. "
            f"ممنوع تسأل عن أي حاجة منها تاني.]\n"
        )

    messages.append({
        "role": "user",
        "content": f"{repeat_warning}البياع: {salesperson_text}\n\n{turn_instruction}",
    })

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