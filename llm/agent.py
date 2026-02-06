# llm/agent.py
"""
LLM Agent using Qwen 2.5 with BitsAndBytes 4-bit quantization.
"""

import torch
import re
import random
from typing import Optional, Generator, List
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from llm.config import MODEL_NAME, BNB_CONFIG, GENERATION_CONFIG
from llm.prompts import build_system_prompt, build_messages

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL (Singleton)
# ══════════════════════════════════════════════════════════════════════════════

_model = None
_tokenizer = None

# Debug flag - set to True to see what's being sent to LLM
DEBUG_PROMPTS = False


def _load_model():
    """Load model once and cache it."""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    print("[LLM] Loading Qwen with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=BNB_CONFIG["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=BNB_CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=BNB_CONFIG["bnb_4bit_use_double_quant"]
    )
    
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"[LLM] ✅ Model loaded on {_model.device}")
    return _model, _tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK RESPONSES (varied, not repetitive)
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_RESPONSES = [
    "ممكن توضحلي أكتر؟",
    "طيب كمّل",
    "وبعدين؟",
    "ماشي، وإيه كمان؟",
    "أوك، فهمت",
    "طيب",
    "تمام",
    "ماشي يعني",
    "أيوه",
    "وإيه تاني؟",
]


def _get_fallback_response() -> str:
    """Get a random fallback response to avoid repetition."""
    return random.choice(FALLBACK_RESPONSES)


def _clean_response(text: str) -> str:
    """Clean response - remove non-Arabic text and keep only Arabic."""
    
    text = text.strip()
    
    # Remove leading question marks
    if text.startswith("؟"):
        text = text[1:].strip()
    
    # Remove common LLM artifacts
    text = re.sub(r'^(العميل:|الزبون:|أنا:|Response:|Customer:)\s*', '', text, flags=re.IGNORECASE)
    
    # Remove words that contain non-Arabic characters
    words = text.split()
    clean_words = []
    
    for word in words:
        # Check if word contains non-Arabic letters
        has_latin = bool(re.search(r'[a-zA-Z]', word))
        has_cyrillic = bool(re.search(r'[\u0400-\u04FF]', word))
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', word))
        has_hebrew = bool(re.search(r'[\u0590-\u05FF]', word))
        
        # Allow Arabic, numbers, and common punctuation
        has_invalid = bool(re.search(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF0-9\s.,،؟!؛:\-()٠-٩]', word))
        
        if has_latin or has_cyrillic or has_chinese or has_hebrew:
            continue
        
        # Skip if mostly invalid characters
        if has_invalid:
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', word))
            if arabic_chars < len(word) / 2:
                continue
        
        clean_words.append(word)
    
    text = ' '.join(clean_words)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove quotes
    text = text.replace('"', '').replace("'", '').replace('"', '').replace('"', '')
    
    # If too short after cleaning, return fallback
    if len(text) < 3:
        return _get_fallback_response()
    
    # Limit length - but be smarter about it
    if len(text) > 250:
        # Try to cut at a natural break
        for sep in ['؟', '.', '!', '،']:
            idx = text[:250].rfind(sep)
            if idx > 50:
                text = text[:idx+1]
                break
        else:
            text = text[:250]
    
    return text.strip()


def _clean_key_points(text: str) -> List[str]:
    """Extract key points from LLM response and return as list."""
    
    # Split by common separators
    lines = re.split(r'[\n\-\•\*\d+\.\)]', text)
    
    key_points = []
    for line in lines:
        # Clean the line
        line = line.strip()
        line = re.sub(r'^[\-\•\*\d+\.\)]+', '', line).strip()
        
        # Skip empty or too short
        if len(line) < 5:
            continue
        
        # Skip non-Arabic
        if not re.search(r'[\u0600-\u06FF]', line):
            continue
        
        # Clean non-Arabic characters
        clean_line = _clean_response(line)
        if len(clean_line) >= 5:
            key_points.append(clean_line)
    
    # Return at least something
    if not key_points:
        return ["محادثة عامة"]
    
    # Limit to 5 key points
    return key_points[:5]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(
    customer_text: str,
    emotion: dict,
    emotional_context: dict,
    persona: dict,
    memory: dict,
    rag_context: dict
) -> str:
    """Generate customer response using LLM."""
    
    model, tokenizer = _load_model()
    
    # Build prompt
    system_prompt = build_system_prompt(persona, emotion, emotional_context, rag_context)
    messages = build_messages(system_prompt, memory, customer_text)
    
    # Debug: Print what we're sending to the LLM
    if DEBUG_PROMPTS:
        print("\n" + "="*60)
        print("[DEBUG] MESSAGES BEING SENT TO LLM:")
        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"  {i+1}. [{role}]: {content}")
        print("="*60 + "\n")
    
    # Tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Increased from 60
            temperature=0.6,     # Slightly lower for more consistency
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2  # Increased to reduce repetition
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = _clean_response(response)
        
        return response
        
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return _get_fallback_response()


def generate_response_streaming(
    customer_text: str,
    emotion: dict,
    emotional_context: dict,
    persona: dict,
    memory: dict,
    rag_context: dict
) -> Generator[str, None, None]:
    """
    Generate response with TRUE token-by-token streaming.
    Yields complete sentences as they form.
    """
    model, tokenizer = _load_model()

    # Build prompt (same as non-streaming)
    system_prompt = build_system_prompt(persona, emotion, emotional_context, rag_context)
    messages = build_messages(system_prompt, memory, customer_text)
    
    # Debug: Print what we're sending to the LLM
    if DEBUG_PROMPTS:
        print("\n" + "="*60)
        print("[DEBUG] STREAMING - MESSAGES BEING SENT TO LLM:")
        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"  {i+1}. [{role}]: {content}")
        print("="*60 + "\n")
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Generation kwargs - UPDATED PARAMS
    gen_kwargs = {
        **inputs,
        "max_new_tokens": 150,       # Increased from 60
        "temperature": 0.6,          # Slightly lower
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,   # Increased
        "streamer": streamer,
    }

    # Run generation in background thread
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Buffer tokens and yield at sentence boundaries
    buffer = ""
    sentence_endings = ('.', '،', '؟', '!', '؛', '?', ',')
    yielded_count = 0

    try:
        for token_text in streamer:
            buffer += token_text

            # Check for sentence boundary
            for i, char in enumerate(buffer):
                if char in sentence_endings:
                    sentence = buffer[:i+1].strip()
                    buffer = buffer[i+1:].strip()

                    if sentence and len(sentence) > 3:
                        cleaned = _clean_response(sentence)
                        if cleaned and len(cleaned) > 3:
                            yielded_count += 1
                            yield cleaned
                    break

        # Yield remaining buffer
        if buffer.strip():
            cleaned = _clean_response(buffer.strip())
            if cleaned and len(cleaned) > 3:
                yielded_count += 1
                yield cleaned
        
        # If nothing was yielded, yield a fallback
        if yielded_count == 0:
            yield _get_fallback_response()

    except Exception as e:
        print(f"[LLM] Streaming error: {e}")
        yield _get_fallback_response()

    thread.join(timeout=10)


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY CHECKPOINT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def summarize_conversation(messages: list) -> str:
    """
    Summarize conversation for memory checkpoints.
    """
    model, tokenizer = _load_model()
    
    # Format conversation
    conv_text = "\n".join([
        f"{'البائع' if m.get('speaker') == 'salesperson' else 'العميل'}: {m.get('text', '')}"
        for m in messages
    ])
    
    # Build prompt for summarization
    prompt = f"""لخص المحادثة التالية في جملة أو جملتين بالعربي المصري:

{conv_text}

الملخص:"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return _clean_response(response)
        
    except Exception as e:
        print(f"[LLM] Summarization error: {e}")
        return "محادثة بيع عقارات"


def extract_key_points(messages: list) -> List[str]:
    """
    Extract key points from conversation for memory checkpoints.
    """
    model, tokenizer = _load_model()
    
    # Format conversation
    conv_text = "\n".join([
        f"{'البائع' if m.get('speaker') == 'salesperson' else 'العميل'}: {m.get('text', '')}"
        for m in messages
    ])
    
    # Build prompt for key points extraction
    prompt = f"""استخرج النقاط المهمة من المحادثة دي (كل نقطة في سطر):

{conv_text}

النقاط المهمة:
-"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return _clean_key_points(response)
        
    except Exception as e:
        print(f"[LLM] Key points extraction error: {e}")
        return ["محادثة عامة"]


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    
    # Enable debug for testing
    DEBUG_PROMPTS = True
    
    print("Testing LLM Agent with BitsAndBytes...")
    
    test_emotion = {"primary_emotion": "neutral", "confidence": 0.8, "intensity": "medium", "scores": {}}
    test_context = {"current": test_emotion, "trend": "stable", "recommendation": "be_professional", "risk_level": "medium"}
    test_persona = {
        "id": "difficult_customer", "name": "عميل صعب", "name_en": "Difficult Customer",
        "personality_prompt": "أنت عميل مصري متشكك وبتفاصل كتير في السعر", "difficulty": "hard",
        "traits": ["متشكك", "بيفاصل"], "default_emotion": "frustrated"
    }
    
    # Simulate a conversation with memory
    test_memory = {
        "session_id": "test",
        "checkpoints": [],
        "recent_messages": [
            {"speaker": "salesperson", "text": "السلام عليكم، معاك أحمد من شركة العقارات"},
            {"speaker": "vc", "text": "وعليكم السلام، أنا بدور على شقة"},
            {"speaker": "salesperson", "text": "عندنا شقة 100 متر في مدينة نصر"},
            {"speaker": "vc", "text": "طيب والسعر كام؟"},
            {"speaker": "salesperson", "text": "السعر مليون جنيه"},
            {"speaker": "vc", "text": "والمقدم كام؟"},
        ],
        "total_turns": 3
    }
    
    test_rag = {"query": "", "documents": [], "total_found": 0}
    
    print("\n" + "="*60)
    print("Testing with conversation history (should NOT ask about price again)")
    print("="*60)
    
    # Salesperson already told price, LLM should NOT ask about it
    test_input = "المقدم 100 ألف جنيه والباقي تقسيط"
    
    print(f"\nالبائع: {test_input}")
    start = time.time()
    response = generate_response(test_input, test_emotion, test_context, test_persona, test_memory, test_rag)
    elapsed = time.time() - start
    print(f"العميل: {response}")
    print(f"⏱️ Time: {elapsed:.2f}s")
    
    print("\n✅ Test completed!")