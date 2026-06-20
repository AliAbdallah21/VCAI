import os
import json
import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/api/chat", tags=["chatbot"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro")

SYSTEM_PROMPT = """You are the VCAI Assistant — a friendly, knowledgeable AI helper for the VCAI platform website.

VCAI is an Egyptian-Arabic AI sales training platform built as a Computer Science graduation project at Misr International University (MIU), Class of 2026.

## What VCAI does
- Provides AI-powered practice conversations for real-estate sales agents in Egyptian Arabic
- The virtual AI customer speaks Egyptian Arabic using a fine-tuned Chatterbox TTS model
- Detects the salesperson's emotions in real time (dual-modal: voice tone + text sentiment)
- Remembers previous training sessions (cross-session memory via PostgreSQL checkpoints)
- Scores each session across 8 sales skills: Rapport Building, Product Knowledge, Needs Assessment, Objection Handling, Communication Clarity, Negotiation, Closing Technique, Follow-up
- Generates automated coaching reports with personalised feedback after every session
- Manager dashboard for team leaders to track all agents' progress

## Technology
- STT: Faster-Whisper (GPU, <450ms latency)
- TTS: Chatterbox fine-tuned on Egyptian Arabic (~2.5s first audio)
- Emotion: emotion2vec + AraBERT fusion
- LLM: OpenRouter (Claude Haiku) / Qwen 2.5 local
- Evaluation: LangGraph two-pass pipeline with FAISS RAG fact-checking
- Backend: FastAPI + PostgreSQL
- Frontend: React + Vite

## Pricing plans
- Free: 1 seat, unlimited sessions
- Starter: 5 seats
- Professional: 20 seats
- Enterprise: Custom seats and SLA

## Team
- Ali Abdallah — AI Pipeline & Backend
- Bakr — TTS & Voice Fine-tuning
- Ismail — LLM & Evaluation
- Menna — Emotion Detection
- Supervisor: Dr. Ahmed Mansour
- T.A.: Karim Mohamed
- University: Misr International University, Cairo, Egypt
- Contact: gradproject11234@gmail.com

## Your behaviour
- Be concise, warm, and helpful
- Answer in the same language the user writes in (Arabic or English)
- If asked about something outside VCAI, politely redirect to what you know
- For enterprise or research enquiries, direct them to the Contact page or email
- Never invent features that don't exist above
"""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


@router.post("/widget")
async def chat_widget(req: ChatRequest):
    if not OPENROUTER_API_KEY:
        return {"error": "Chat service not configured"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages[-12:]]

    def stream():
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": OPENROUTER_MODEL, "messages": messages, "max_tokens": 400, "stream": True},
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                stream=True, timeout=30,
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
