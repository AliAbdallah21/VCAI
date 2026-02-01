# llm/agent.py
"""
LLM Agent using Qwen 2.5 with BitsAndBytes 4-bit quantization.
"""

import torch
import re
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


def _clean_response(text: str) -> str:
    """Clean response - remove non-Arabic text and keep only Arabic."""
    
    text = text.strip()
    
    # Remove leading question marks
    if text.startswith("؟"):
        text = text[1:].strip()
    
    # Remove words that contain non-Arabic characters
    words = text.split()
    clean_words = []
    
    for word in words:
        # Check if word contains non-Arabic letters
        has_latin = bool(re.search(r'[a-zA-Z]', word))
        has_cyrillic = bool(re.search(r'[\u0400-\u04FF]', word))
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', word))
        has_hebrew = bool(re.search(r'[\u0590-\u05FF]', word))
        has_other = bool(re.search(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF0-9\s.,،؟!؛:\-()]', word))
        
        if has_latin or has_cyrillic or has_chinese or has_hebrew or has_other:
            continue
        
        clean_words.append(word)
    
    text = ' '.join(clean_words)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove quotes
    text = text.replace('"', '').replace("'", '').replace('"', '').replace('"', '')
    
    # If too short after cleaning, return fallback
    if len(text) < 5:
        return "طيب، والسعر كام؟"  # Customer-like fallback
    
    # Limit length
    if len(text) > 200:
        for sep in ['،', '.', '؟', '!']:
            idx = text[:200].rfind(sep)
            if idx > 50:
                text = text[:idx+1]
                break
        else:
            text = text[:200]
    
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
    
    # Tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = _clean_response(response)
        
        return response
        
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return "ممكن تكرر تاني؟"


# ══════════════════════════════════════════════════════════════════════════════
# PATCH: Replace generate_response_streaming() in C:\VCAI\llm\agent.py
# Find the existing generate_response_streaming function and replace it with this
# ══════════════════════════════════════════════════════════════════════════════

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
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Generation kwargs
    gen_kwargs = {
        **inputs,
        "max_new_tokens": 60,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.15,
        "streamer": streamer,
    }

    # Run generation in background thread
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Buffer tokens and yield at sentence boundaries
    buffer = ""
    sentence_endings = ('.', '،', '؟', '!', '؛', '?', ',')

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
                            yield cleaned
                    break

        # Yield remaining buffer
        if buffer.strip():
            cleaned = _clean_response(buffer.strip())
            if cleaned and len(cleaned) > 3:
                yield cleaned

    except Exception as e:
        print(f"[LLM] Streaming error: {e}")
        yield "ممكن تكرر تاني؟"

    thread.join(timeout=10)


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY CHECKPOINT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def summarize_conversation(messages: list) -> str:
    """
    Summarize conversation for memory checkpoints.
    
    Called by memory_node when creating checkpoints every N turns.
    
    Args:
        messages: List of message dicts with 'speaker' and 'text' keys
    
    Returns:
        str: Arabic summary of the conversation
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
    
    Called by memory_node when creating checkpoints.
    
    Args:
        messages: List of message dicts with 'speaker' and 'text' keys
    
    Returns:
        List[str]: List of key points in Arabic
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
    
    print("Testing LLM Agent with BitsAndBytes...")
    
    test_emotion = {"primary_emotion": "frustrated", "confidence": 0.8, "intensity": "medium", "scores": {}}
    test_context = {"current": test_emotion, "trend": "stable", "recommendation": "be_firm", "risk_level": "medium"}
    test_persona = {
        "id": "difficult_customer", "name": "عميل صعب", "name_en": "Difficult Customer",
        "personality_prompt": "أنت عميل مصري متشكك وبتفاصل كتير في السعر", "difficulty": "hard",
        "traits": ["متشكك", "بيفاصل"], "default_emotion": "frustrated"
    }
    test_memory = {"session_id": "test", "checkpoints": [], "recent_messages": [], "total_turns": 0}
    test_rag = {"query": "", "documents": [{"content": "شقة 120 متر في التجمع الخامس بسعر 2,500,000 جنيه، تشطيب سوبر لوكس", "source": "properties.pdf", "score": 0.9}], "total_found": 1}
    
    # Test generate_response
    test_inputs = [
        "السلام عليكم",
        "عندنا شقة حلوة في التجمع",
        "السعر 2 مليون ونص",
    ]
    
    print("\n" + "="*60)
    print("Testing generate_response()")
    print("="*60)
    
    for text in test_inputs:
        print(f"\nالبائع: {text}")
        start = time.time()
        response = generate_response(text, test_emotion, test_context, test_persona, test_memory, test_rag)
        elapsed = time.time() - start
        print(f"العميل: {response}")
        print(f"⏱️ Time: {elapsed:.2f}s")
    
    # Test summarize_conversation
    print("\n" + "="*60)
    print("Testing summarize_conversation()")
    print("="*60)
    
    test_messages = [
        {"speaker": "salesperson", "text": "السلام عليكم، معاك أحمد من شركة العقارات"},
        {"speaker": "vc", "text": "وعليكم السلام، أنا بدور على شقة في التجمع"},
        {"speaker": "salesperson", "text": "عندنا شقة 120 متر بسعر 2 مليون ونص"},
        {"speaker": "vc", "text": "ده غالي أوي، فيه أرخص؟"},
        {"speaker": "salesperson", "text": "ممكن نتكلم في السعر لو جاد"},
    ]
    
    start = time.time()
    summary = summarize_conversation(test_messages)
    elapsed = time.time() - start
    print(f"Summary: {summary}")
    print(f"⏱️ Time: {elapsed:.2f}s")
    
    # Test extract_key_points
    print("\n" + "="*60)
    print("Testing extract_key_points()")
    print("="*60)
    
    start = time.time()
    key_points = extract_key_points(test_messages)
    elapsed = time.time() - start
    print(f"Key Points:")
    for i, point in enumerate(key_points, 1):
        print(f"  {i}. {point}")
    print(f"⏱️ Time: {elapsed:.2f}s")
    
    print("\n✅ All tests completed!")