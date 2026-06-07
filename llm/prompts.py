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

from typing import Optional

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


def _format_scenario_block(scenario: dict) -> str:
    """
    Render the buyer scenario as an Arabic prompt section. This is the
    customer's HARD situation for this call — budget ceiling, timeline,
    must-haves, deal-breakers — and overrides any generic numbers the
    persona prompt might mention.
    """
    if not scenario:
        return ""

    lines = ["موقفك الحقيقي في المكالمة دي (ده وضعك — التزم بيه في كل قراراتك):"]

    label = scenario.get("label")
    if label:
        lines.append(f"- {label}")

    bmin = scenario.get("budget_min")
    bmax = scenario.get("budget_max")
    if isinstance(bmin, int) and isinstance(bmax, int):
        lines.append(
            f"- ميزانيتك من {bmin:,} لـ {bmax:,} جنيه. "
            f"لو السعر النهائي عدّى {bmax:,} جنيه إنت مش هتشتري."
        )

    must_haves = scenario.get("must_haves") or []
    if must_haves:
        lines.append("- لازم يكون فيه: " + "، ".join(must_haves))

    deal_breakers = scenario.get("deal_breakers") or []
    if deal_breakers:
        lines.append("- هتمشي وما تشتريش لو: " + "، ".join(deal_breakers))

    lines.append("لما تقرر توافق أو ترفض، اعتمد على الموقف ده — مش على أرقام عامة.")
    return "\n".join(lines)


_TRAINING_FOCUS_BLOCKS: dict[str, str] = {
    "closing": (
        "[توجيه تدريبي — التركيز على الإغلاق]\n"
        "أنت مهتم بالشقة لكن مش ملتزم. خلّق فرص إغلاق طبيعية:\n"
        "- قول حاجات زي «أنا محتاج أفكر» أو «مش عارف لو هحجز دلوقتي»\n"
        "- لو البياع حاول يقفل الصفقة، ما توافقش بسهولة — اطلب منه سبب محدد ليه تحجز دلوقتي\n"
        "- اسأل عن خطوات الحجز والإجراءات لكن من غير ما تلتزم"
    ),
    "objection_handling": (
        "[توجيه تدريبي — التركيز على الاعتراضات]\n"
        "ارفع على الأقل 2-3 اعتراضات حقيقية خلال المحادثة:\n"
        "- اعتراض على السعر: «ده غالي أوي عليا»\n"
        "- اعتراض على الموعد: «الاستلام بعد 3 سنين ده وقت طويل»\n"
        "- اعتراض على الشروط: «المقدم كتير، مش قادر أدفعه»\n"
        "- لو البياع رد على اعتراض، اعترض على حاجة تانية"
    ),
    "product_knowledge": (
        "[توجيه تدريبي — التركيز على معرفة المنتج]\n"
        "اسأل أسئلة تفصيلية دقيقة تختبر معرفة البياع:\n"
        "- التشطيب: «السوبر لوكس بتاعكم بيشمل إيه بالظبط؟»\n"
        "- الخدمات: «فيه جيم وحمام سباحة؟ بيتصرف إزاي؟»\n"
        "- الموقع: «البُعد عن الرينج روود كام كيلو بالظبط؟»\n"
        "- القانوني: «الوحدة مسجلة ولا على عقد ابتدائي؟»\n"
        "- لو البياع ما جاوبش بدقة، ألح في السؤال"
    ),
    "rapport": (
        "[توجيه تدريبي — التركيز على بناء العلاقة]\n"
        "ابدأ المحادثة بارداً ومتحفظاً:\n"
        "- ردود مقتضبة في الأول، مش حماسي\n"
        "- لو البياع كسر الجليد بطريقة اصطناعية، ما تفرهدش فوراً\n"
        "- لو البياع كان صادق ومهتم بيك كشخص مش كعميل فقط، دوّب شوية\n"
        "- الهدف إن البياع يكسب ثقتك بالتدريج، مش في لحظة"
    ),
    "communication": (
        "[توجيه تدريبي — التركيز على التواصل]\n"
        "اتكلم بسرعة وبتشعّب:\n"
        "- اسأل أكتر من سؤال في وقت واحد\n"
        "- لو البياع ما اتكلمش بوضوح، قول «مش فاهم قصدك إيه»\n"
        "- ما تكملش حتى ما يجاوب على كل نقطة قلتها\n"
        "- لو البياع اتكلم كتير من غير ما يوصل لنقطة، قول «ممكن تلخصلي في جملتين؟»"
    ),
}


def build_system_prompt(
    persona: dict,
    emotion: dict,
    emotional_context: dict,
    rag_context: dict,
    training_focus: Optional[str] = None,
) -> str:
    """Build system prompt for the LLM."""

    persona_name = persona.get("name", persona.get("name_ar", "عميل"))
    personality = persona.get("personality_prompt", "أنت عميل مصري بتدور على شقة")
    difficulty = persona.get("difficulty", "medium")
    traits = persona.get("traits", [])
    scenario_block = _format_scenario_block(persona.get("scenario"))
    gender = persona.get("gender", "male")

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

    emotion_guidance = EMOTION_GUIDANCE.get(current_emotion, "انت هادي")

    if gender == "female":
        gender_preamble = f"أنتِ عميلة مصرية اسمك {persona_name} بتدوري على شقة تشتريها. أنتِ العميلة مش البياع."
        difficulty_guidance = {
            "hard": "أنتِ عميلة صعبة، بتفاصلي في السعر وشكّاكة جداً. اسأل أسئلة صعبة.",
            "easy": "أنتِ عميلة سهلة ومتعاونة. بتسأل أسئلة بسيطة.",
            "medium": "أنتِ عميلة عادية، بتسأل أسئلة منطقية.",
        }.get(difficulty, "أنتِ عميلة عادية")
    else:
        gender_preamble = f"أنت عميل مصري اسمك {persona_name} بتدور على شقة تشتريها. أنت العميل مش البياع."
        difficulty_guidance = {
            "hard": "انت عميل صعب، بتفاصل في السعر وشكاك جداً. اسأل أسئلة صعبة.",
            "easy": "انت عميل سهل ومتعاون. بتسأل أسئلة بسيطة.",
            "medium": "انت عميل عادي، بتسأل أسئلة منطقية.",
        }.get(difficulty, "انت عميل عادي")

    # Build examples string
    examples_str = "\n".join([f'- "{ex}"' for ex in EGYPTIAN_EXAMPLES[:10]])

    scenario_section = f"\n{scenario_block}\n" if scenario_block else ""
    focus_block = (
        f"\n\n{_TRAINING_FOCUS_BLOCKS[training_focus]}"
        if training_focus and training_focus in _TRAINING_FOCUS_BLOCKS
        else ""
    )

    return f"""{gender_preamble}

شخصيتك: {personality}
حالتك: {emotion_guidance}
{difficulty_guidance}
{scenario_section}
اللهجة: عامية مصرية قاهرية فقط. قول "إيه/كام/فين/عايز/مفيش/ده/كده" مش الفصحى.

معلومات العقارات: {rag_info if rag_info else "اسأل البياع عن التفاصيل"}

أمثلة ردود صح:
- "مليون ونص ده غالي شوية. فيه حاجة أرخص في نفس المنطقة؟"
- "التجمع كويسة! المواصلات إيه هناك؟"
- "وعليكم السلام. أنا بدور على شقة 3 غرف، عندك إيه؟"
- "120 متر حلو. الدور إيه؟"

ردودك لازم تكون متعلقة بالمحادثة، مش ردود عامة.{focus_block}"""


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


# ──────────────────────────────────────────────────────────────────────────────
# Customer journey stages
#
# A real cold-call customer doesn't start in "interview mode" — they warm up:
#   cold       -> doesn't know who is calling or why
#   screening  -> knows it's a real-estate pitch, deciding whether to give time
#   engaged    -> willing to listen, asking basic questions
#   evaluating -> actively weighing the offer against their criteria
#   decision   -> the seller has invited closure; buy / walk / defer
#
# The stage is derived by a cheap heuristic (no extra LLM call) from signals we
# already track: how many turns in, how many topics covered, whether the seller
# has identified themselves, and whether the seller is explicitly closing.
# ──────────────────────────────────────────────────────────────────────────────

# Cues that the salesperson has established WHO they are / WHY they're calling.
# Once any of these appears, the customer is no longer "cold" — they know this
# is a real-estate sales contact.
_INTRO_CUES = [
    "شركة", "عقار", "عقارات", "سمسار", "مستشار عقاري", "تطوير عقاري",
    "تسويق عقاري", "كمبوند", "مشروع", "وحدة سكنية", "عرض عقاري",
    "حسن علام", "بكلمك من", "معاك من", "اسمي", "أنا من",
]


def _count_salesperson_turns(memory: dict) -> int:
    """How many times the salesperson has spoken so far (recent + checkpointed)."""
    n = 0
    for m in memory.get("recent_messages", []) or []:
        if m.get("speaker") == "salesperson":
            n += 1
    # Each checkpoint compresses ~5 turns that have rolled out of the window.
    n += len(memory.get("checkpoints", []) or []) * 5
    return n


def _seller_introduced(memory: dict, salesperson_text: str) -> bool:
    """
    True once the salesperson has identified themselves or their purpose.
    Scans the current message + all prior salesperson messages. A checkpointed
    conversation is deep enough that the intro definitely already happened.
    """
    if memory.get("checkpoints"):
        return True
    blob_parts = [salesperson_text or ""]
    for m in memory.get("recent_messages", []) or []:
        if m.get("speaker") == "salesperson":
            blob_parts.append(m.get("text", ""))
    blob = " ".join(blob_parts).lower()
    return any(cue in blob for cue in _INTRO_CUES)


def detect_journey_stage(
    memory: dict,
    salesperson_text: str,
    topic_count: int,
    seller_closing: bool,
) -> str:
    """
    Derive the current customer-journey stage. Pure heuristic — no LLM call.

    Returns one of: 'cold', 'screening', 'engaged', 'evaluating', 'decision'.
    """
    # Seller explicitly inviting closure overrides everything.
    if seller_closing:
        return "decision"

    introduced = _seller_introduced(memory, salesperson_text)
    sp_turns = _count_salesperson_turns(memory) + 1  # +1 for the current turn

    # Still cold: salesperson hasn't said who they are / why, and it's early.
    if not introduced and sp_turns <= 2:
        return "cold"

    # Knows it's a pitch but barely anything concrete discussed yet.
    if topic_count <= 1:
        return "screening"

    # Some basics covered — actively listening and asking.
    if topic_count <= 4:
        return "engaged"

    # 5+ topics covered — weighing the offer, close to a decision.
    return "evaluating"


# How hard the customer is to move through the journey. This is the SESSION
# difficulty — orthogonal to persona. It controls the *friction* of advancing:
# an easy customer forgives a weak opening and decides fast; a hard one demands
# proof at every stage and can regress or hang up.
_DIFFICULTY_MODIFIER = {
    "easy": (
        "أنت عميل متعاون ومتسامح: لو بداية البياع ضعيفة سامحه واكمل، اتقدّم في "
        "المحادثة بسهولة، وادي البياع فرصة. لو العرض كويس قرّر بسرعة من غير "
        "تعقيد."
    ),
    "medium": (
        "أنت عميل عادي: اتعامل بمنطق، اتقدّم في المحادثة لما البياع يقدّملك سبب "
        "كويس، واسأل أسئلة معقولة قبل ما تقرّر."
    ),
    "hard": (
        "أنت عميل صعب وشكّاك: متشكّك في كل مرحلة، اطلب إثبات وتفاصيل قبل ما "
        "تتقدّم، ما تتحمّسش بسهولة. لو البياع وعد بحاجة وما وضّحهاش، أو رد بطريقة "
        "وحشة أو ضغط عليك، ارجع خطوة لورا في حذرك أو هدّد إنك هتنهي المكالمة. "
        "إقفال الصفقة معاك لازم يتعب البياع."
    ),
}


def _build_turn_instruction(
    stage: str,
    topic_count: int,
    difficulty: str = "medium",
) -> str:
    """
    Build the per-turn instruction for the customer, tailored to the current
    journey stage AND the session difficulty. Earlier stages stay guarded;
    later stages push toward a decision. Difficulty controls how much friction
    the salesperson meets when trying to advance the customer.
    """
    if stage == "cold":
        core = (
            "مرحلة المحادثة: البياع لسه ما عرّفش نفسه كويس وأنت مش متأكد مين ده "
            "ولا إيه الموضوع.\n"
            "• رد بحذر طبيعي زي أي حد جاله اتصال مش متوقع: اسأل 'مين معايا؟' أو "
            "'الموضوع إيه؟' أو 'حضرتك بتكلمني في إيه؟'\n"
            "• ممنوع تبدأ تسأل عن شقق أو أسعار أو تفاصيل — أنت لسه مش عارف ده إيه أصلاً.\n"
            "• لو حسيت إن البياع بيضغط أو وحش، تقدر تنهي المكالمة بأدب.\n"
            "الرد جملة قصيرة واحدة."
        )
    elif stage == "screening":
        core = (
            "مرحلة المحادثة: البياع عرّف نفسه وعرفت إنه عرض عقاري. أنت لسه بتقرر "
            "تدّيله وقت ولا لأ.\n"
            "• رد بحذر متحفّظ — مش متحمّس بدري. اسأل سؤال يخلّيه يوضّح الفرصة "
            "(زي 'طيب عندك إيه بالظبط؟' أو 'وليه أهتم؟').\n"
            "• لو البياع بيضغط أو كلامه فاضي، قول إنك مشغول ممكن تنهي المكالمة.\n"
            "الرد جملة لجملتين قصيرين."
        )
    elif stage == "engaged":
        core = (
            "مرحلة المحادثة: قررت تسمع العرض. ابدأ تجمع المعلومات.\n"
            "• علّق على اللي قاله البياع، واسأل سؤال جديد عن حاجة أساسية تهمّك "
            "(السعر، المكان، المساحة، التقسيط...) لسه ما اتكلمتش عنها.\n"
            "• لو لقيت إن العرض كله عاجبك حسب 'متى توافق': تقدر تقفل الصفقة.\n"
            "• لو في حاجة فوق ميزانيتك أو غير واضحة: اعتذر بأدب وامشي.\n"
            "الرد جملة لجملتين بحد أقصى. ممنوع ترد بكلمة واحدة لوحدها."
        )
    elif stage == "evaluating":
        core = (
            f"مرحلة المحادثة: اتغطّى {topic_count} موضوع بالفعل — أنت بتقيّم العرض "
            "وقربت تاخد قرار. اختار واحد:\n"
            "1. لو المعلومات كافية حسب شخصيتك (راجع 'متى توافق'): اقفل الصفقة. "
            "قول 'تمام، أنا موافق' أو 'يلا نتفق'.\n"
            "2. لو لقيت حاجة فوق حدودك (راجع 'متى ترفض'): اعتذر بأدب وامشي.\n"
            "3. لو فعلاً ضروري سؤال أخير واحد بس عن حاجة لسه ما اتغطّتش: اسأل، "
            "بس بعدها لازم تقرر في الرد اللي بعده.\n"
            "ممنوع تكرر نفس النوع من الأسئلة. الرد جملة لجملتين بحد أقصى."
        )
    else:  # stage == "decision"
        core = (
            "مرحلة المحادثة: البياع بيعرض عليك تقفل الصفقة دلوقتي. لازم تختار:\n"
            "1. لو راضي بالعرض حسب معايير شخصيتك (متى توافق): قول 'تمام، أنا موافق' "
            "أو 'يلا نمشي للعقد'.\n"
            "2. لو في حاجة مش مناسبة (متى ترفض): قول 'شكراً، هفكر تاني' أو "
            "'العرض ده مش هياخده'.\n"
            "ممنوع تسأل سؤال جديد. لازم تقرر دلوقتي. الرد جملة لجملتين بحد أقصى."
        )

    diff_line = _DIFFICULTY_MODIFIER.get(difficulty, _DIFFICULTY_MODIFIER["medium"])
    return f"[طبعك في المحادثة دي: {diff_line}\n\n{core}]"


def build_messages(
    system_prompt: str,
    memory: dict,
    salesperson_text: str,
    difficulty: str = "medium",
) -> list:
    """
    Build message list for chat completion.

    IMPORTANT: Includes checkpoint summaries + recent messages
    so the LLM remembers the full conversation context.

    Args:
        difficulty: session difficulty (easy/medium/hard) — controls how much
            friction the customer puts up at each journey stage.
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
    # 3. JOURNEY-STAGE-AWARE PER-TURN INSTRUCTION
    # The instruction adapts to where the customer is in the journey — cold
    # (doesn't know who's calling) -> screening -> engaged -> evaluating ->
    # decision. Earlier stages stay guarded; later stages push for a decision.
    # Putting this at the decision point (not just in the system prompt) is
    # essential — with streaming, the model commits to a path before fully
    # consuming a long system prompt.
    # ══════════════════════════════════════════════════════════════════════════
    already_mentioned = _extract_already_mentioned(memory)
    seller_closing = _seller_is_closing(salesperson_text)
    journey_stage = detect_journey_stage(
        memory, salesperson_text, len(already_mentioned), seller_closing
    )
    turn_instruction = _build_turn_instruction(
        journey_stage, len(already_mentioned), difficulty
    )

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