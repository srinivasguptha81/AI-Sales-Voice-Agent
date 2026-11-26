from __future__ import annotations

from typing import Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.agent import SalesVoiceAgent
from app.config import KB_DIR, STATIC_DIR, get_settings
from app.crm import CRMStore
from app.models import CallSummaryResponse, ChatRequest, ChatResponse, TTSRequest, to_dict
from app.rag import KnowledgeBaseRAG

settings = get_settings()
rag = KnowledgeBaseRAG(KB_DIR)
crm = CRMStore(settings.database_path)
agent = SalesVoiceAgent(settings, rag, crm)

app = FastAPI(
    title="Fortivex Autonomous AI Sales Voice Agent",
    description="Real-time multilingual sales voice agent with RAG, BANT qualification, sentiment analysis, and CRM booking.",
    version="1.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "active_provider": settings.active_provider,
        "kb_chunks": len(rag.chunks),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_turn(req: ChatRequest) -> ChatResponse:
    return await agent.handle_turn(req)


@app.post("/api/sessions/{session_id}/summary", response_model=CallSummaryResponse)
async def end_call_summary(session_id: str) -> CallSummaryResponse:
    return agent.summarize_call(session_id)


@app.get("/api/kb/search")
async def kb_search(q: str, top_k: int = 3) -> dict:
    return {"query": q, "results": [to_dict(c) for c in rag.search(q, top_k=top_k)]}


@app.get("/api/crm/leads")
async def list_leads() -> dict:
    return {"leads": crm.list_leads()}


@app.get("/api/crm/appointments")
async def list_appointments() -> dict:
    return {"appointments": crm.list_appointments()}


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)) -> dict:
    """Server-side STT via OpenAI Whisper when OPENAI_API_KEY is configured."""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=501,
            detail="Server-side STT requires OPENAI_API_KEY; use browser Web Speech API in the UI.",
        )
    content = await audio.read()
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            data={"model": "whisper-1"},
            files={"file": (audio.filename or "audio.webm", content, audio.content_type or "audio/webm")},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest) -> Response:
    """Server-side TTS via OpenAI TTS when OPENAI_API_KEY is configured."""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=501,
            detail="Server-side TTS requires OPENAI_API_KEY; use browser SpeechSynthesis in the UI.",
        )
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": "tts-1", "voice": req.voice, "input": req.text},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return Response(content=resp.content, media_type="audio/mpeg")
