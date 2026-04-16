# llm/agent.py
"""
LLM Agent with switchable backend:
- Qwen 2.5 (local, with BitsAndBytes 4-bit quantization)
- OpenRouter API (cloud, supports Claude/GPT/etc.)

Set USE_OPENROUTER=true in .env to use OpenRouter instead of local Qwen.
"""


import os
from dotenv import load_dotenv
load_dotenv()
import json
import queue as _queue
import torch
import re
import random
import requests
from typing import Optional, Generator, List
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from llm.config import MODEL_NAME, BNB_CONFIG, GENERATION_CONFIG
from llm.prompts import build_system_prompt, build_messages

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Switch between Qwen (local) and OpenRouter (cloud)
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

# Debug flag - set to True to see full message content sent to LLM
DEBUG_PROMPTS = True

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL (Singleton) - Only used for Qwen
# ══════════════════════════════════════════════════════════════════════════════

_model = None
_tokenizer = None


def _load_model():
    """Load Qwen model once and cache it. Only called if USE_OPENROUTER=false."""
    global _model, _tokenizer
    
    if USE_OPENROUTER:
        print("[LLM] Using OpenRouter API - skipping local model load")
        return None, None
    
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
# OPENROUTER API CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class OpenRouterClient:
    """Client for OpenRouter API - supports Claude, GPT, etc."""
    
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = OPENROUTER_MODEL
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        
        print(f"[LLM] OpenRouter client initialized with model: {self.model}")
    
    def generate(self, messages: List[dict], **kwargs) -> str:
        """Generate response using OpenRouter API."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 300),
        }
        
        if DEBUG_PROMPTS:
            print(f"[LLM] OpenRouter request to {self.model}")
            for msg in messages:
                preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"  [{msg['role']}]: {preview}")
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"]
            
            if DEBUG_PROMPTS:
                print(f"[LLM] OpenRouter response: {result[:100]}...")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"[LLM] OpenRouter API error: {e}")
            raise


# Singleton OpenRouter client
_openrouter_client = None

def _get_openrouter_client() -> OpenRouterClient:
    """Get or create OpenRouter client."""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client


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
    """
    Light-touch cleanup: strip role prefixes, remove quotes, limit length.
    Never filters by character set — trust the LLM to produce Arabic.
    """
    raw_len = len(text)
    text = text.strip()

    # Remove leading question marks
    if text.startswith("؟"):
        text = text[1:].strip()

    # Remove common LLM role prefixes
    text = re.sub(
        r'^(العميل:|الزبون:|أنا:|Response:|Customer:)\s*',
        '', text, flags=re.IGNORECASE
    )

    # Remove curly/straight quotes
    text = text.replace('"', '').replace('"', '').replace('"', '').replace("'", '')

    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Truncate at a natural sentence boundary if very long
    if len(text) > 300:
        for sep in ('؟', '!', '.', '،'):
            idx = text[:300].rfind(sep)
            if idx > 50:
                text = text[:idx + 1]
                break
        else:
            text = text[:300]

    if not text:
        return _get_fallback_response()

    return text


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
    """Generate customer response using LLM (Qwen or OpenRouter)."""
    
    # Build prompt
    system_prompt = build_system_prompt(persona, emotion, emotional_context, rag_context)
    messages = build_messages(system_prompt, memory, customer_text)
    
    # Debug: Print what we're sending to the LLM
    if DEBUG_PROMPTS:
        print("\n" + "="*60)
        print(f"[DEBUG] MESSAGES BEING SENT TO LLM (USE_OPENROUTER={USE_OPENROUTER}):")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  {i+1}. [{role}]: {content_preview}")
        print("="*60 + "\n")
    
    try:
        if USE_OPENROUTER:
            # Use OpenRouter API
            client = _get_openrouter_client()
            response = client.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            response = _clean_response(response)
            return response
        else:
            # Use local Qwen model
            model, tokenizer = _load_model()
            
            # Tokenize
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = _clean_response(response)
            return response
        
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return _get_fallback_response()


def _stream_openrouter_sentences(messages: list, **kwargs) -> Generator[str, None, None]:
    """
    Stream sentence-level output from OpenRouter via SSE.

    OpenRouter SSE format:
        data: {"choices":[{"delta":{"content":"token"},...}]}
        data: [DONE]

    Accumulates tokens in a buffer, yields complete sentences as they arrive:
    - Primary split: . ؟ ! followed by space (or end of stream)
    - Secondary split: ، (Arabic comma) after 15+ words (prevents stalling on long clauses)

    Uses a background thread so network I/O doesn't block the calling generator.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 300),
        "stream": True,
    }

    sentence_q: _queue.Queue = _queue.Queue()

    def _worker():
        buffer = ""

        def _flush_sentences(buf: str) -> str:
            """Extract all complete sentences from buf, enqueue them, return remainder."""
            while True:
                # Primary: scan for . ؟ ! followed by space or end-of-buffer
                split_pos = -1
                for idx, ch in enumerate(buf):
                    if ch in ".؟!":
                        after = idx + 1
                        if after >= len(buf) or buf[after] in " \n\t":
                            split_pos = after
                            break

                if split_pos != -1:
                    sentence = buf[:split_pos].strip()
                    if len(sentence) > 3:
                        sentence_q.put(sentence)
                    buf = buf[split_pos:].lstrip()
                    continue  # keep scanning remaining buffer

                # Secondary: ، after 15+ words
                comma_idx = buf.find("،")
                if comma_idx > 0:
                    words_before = len(buf[:comma_idx].split())
                    if words_before >= 15:
                        sentence = buf[:comma_idx + 1].strip()
                        if len(sentence) > 3:
                            sentence_q.put(sentence)
                        buf = buf[comma_idx + 1:].lstrip()
                        continue

                break  # no more boundaries in current buffer

            return buf

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=30,
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"  # force UTF-8 — prevents Arabic mojibake

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data: "):
                    continue

                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    content = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if content:
                        buffer += content
                        buffer = _flush_sentences(buffer)
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

            # Flush anything left after [DONE]
            remaining = buffer.strip()
            if len(remaining) > 3:
                sentence_q.put(remaining)

        except Exception as exc:
            print(f"[LLM] OpenRouter SSE error: {exc}")

        finally:
            sentence_q.put(None)  # sentinel — generator stops here

    thread = Thread(target=_worker, daemon=True)
    thread.start()

    yielded = 0
    while True:
        sentence = sentence_q.get()
        if sentence is None:
            break
        cleaned = _clean_response(sentence)
        if cleaned and len(cleaned) > 3:
            yielded += 1
            yield cleaned

    thread.join(timeout=30)

    if yielded == 0:
        yield _get_fallback_response()


def generate_response_streaming(
    customer_text: str,
    emotion: dict,
    emotional_context: dict,
    persona: dict,
    memory: dict,
    rag_context: dict
) -> Generator[str, None, None]:
    """
    Generate response with true sentence-level streaming.

    OpenRouter: SSE stream → accumulate tokens → yield sentences as they complete.
    Qwen:       TextIteratorStreamer (background thread) → yield sentences.

    First sentence arrives in ~1.5s vs ~5-7s for the full response,
    enabling TTS to start immediately and achieve ~3.5s first-audio latency.
    """

    if USE_OPENROUTER:
        system_prompt = build_system_prompt(persona, emotion, emotional_context, rag_context)
        messages = build_messages(system_prompt, memory, customer_text)

        # Always-on diagnostics so we can verify the prompt is correct
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        print(f"\n[LLM DEBUG] System prompt length: {len(system_prompt)} chars")
        print(f"[LLM DEBUG] Messages count: {len(messages)}")
        print(f"[LLM DEBUG] System prompt preview: {system_prompt[:200]}...")
        print(f"[LLM DEBUG] Last user message: {last_user_msg[:150]}")

        if DEBUG_PROMPTS:
            print("\n" + "=" * 60)
            print("[DEBUG] STREAMING OpenRouter — FULL MESSAGES:")
            for i, msg in enumerate(messages):
                preview = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
                print(f"  {i+1}. [{msg['role']}]: {preview}")
            print("=" * 60 + "\n")

        yield from _stream_openrouter_sentences(messages, temperature=0.7, max_tokens=300)
        return
    
    # Qwen: True token-by-token streaming
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

    # Generation kwargs
    gen_kwargs = {
        **inputs,
        "max_new_tokens": 150,
        "temperature": 0.6,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,
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
    """Summarize conversation for memory checkpoints."""
    
    # Format conversation
    conv_text = "\n".join([
        f"{'البائع' if m.get('speaker') == 'salesperson' else 'العميل'}: {m.get('text', '')}"
        for m in messages
    ])
    
    if USE_OPENROUTER:
        # Use OpenRouter
        client = _get_openrouter_client()
        prompt_messages = [
            {"role": "system", "content": "أنت مساعد بيلخص المحادثات بالعربي المصري. لخص في جملة أو جملتين بس."},
            {"role": "user", "content": f"لخص المحادثة دي:\n\n{conv_text}"}
        ]
        try:
            response = client.generate(prompt_messages, temperature=0.5, max_tokens=100)
            return _clean_response(response)
        except Exception as e:
            print(f"[LLM] Summarization error: {e}")
            return "محادثة بيع عقارات"
    else:
        # Use Qwen
        model, tokenizer = _load_model()
        
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
    """Extract key points from conversation for memory checkpoints."""
    
    # Format conversation
    conv_text = "\n".join([
        f"{'البائع' if m.get('speaker') == 'salesperson' else 'العميل'}: {m.get('text', '')}"
        for m in messages
    ])
    
    if USE_OPENROUTER:
        # Use OpenRouter
        client = _get_openrouter_client()
        prompt_messages = [
            {"role": "system", "content": "استخرج النقاط المهمة من المحادثة. اكتب كل نقطة في سطر منفصل."},
            {"role": "user", "content": f"المحادثة:\n\n{conv_text}\n\nالنقاط المهمة:"}
        ]
        try:
            response = client.generate(prompt_messages, temperature=0.5, max_tokens=150)
            return _clean_key_points(response)
        except Exception as e:
            print(f"[LLM] Key points extraction error: {e}")
            return ["محادثة عامة"]
    else:
        # Use Qwen
        model, tokenizer = _load_model()
        
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
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_llm_info() -> dict:
    """Get information about the current LLM configuration."""
    return {
        "backend": "openrouter" if USE_OPENROUTER else "qwen",
        "model": OPENROUTER_MODEL if USE_OPENROUTER else MODEL_NAME,
        "loaded": _model is not None if not USE_OPENROUTER else True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    
    # Enable debug for testing
    DEBUG_PROMPTS = True
    
    print(f"Testing LLM Agent (USE_OPENROUTER={USE_OPENROUTER})...")
    print(f"LLM Info: {get_llm_info()}")
    
    test_emotion = {"primary_emotion": "neutral", "confidence": 0.8, "intensity": "medium", "scores": {}}
    test_context = {"current": test_emotion, "trend": "stable", "recommendation": "be_professional", "risk_level": "medium"}
    test_persona = {
        "id": "difficult_customer", "name": "عميل صعب", "name_en": "Difficult Customer",
        "personality_prompt": "أنت عميل مصري متشكك وبتفاصل كتير في السعر", "difficulty": "hard",
        "traits": ["متشكك", "بيفاصل"], "default_emotion": "frustrated"
    }
    
    test_memory = {
        "session_id": "test",
        "checkpoints": [],
        "recent_messages": [
            {"speaker": "salesperson", "text": "السلام عليكم، معاك أحمد من شركة العقارات"},
            {"speaker": "vc", "text": "وعليكم السلام، أنا بدور على شقة"},
        ],
        "total_turns": 1
    }
    
    test_rag = {"query": "", "documents": [], "total_found": 0}
    
    print("\n" + "="*60)
    print("Testing conversation response")
    print("="*60)
    
    test_input = "عندنا شقة 100 متر في مدينة نصر"
    
    print(f"\nالبائع: {test_input}")
    start = time.time()
    response = generate_response(test_input, test_emotion, test_context, test_persona, test_memory, test_rag)
    elapsed = time.time() - start
    print(f"العميل: {response}")
    print(f"⏱️ Time: {elapsed:.2f}s")
    
    print("\n✅ Test completed!")