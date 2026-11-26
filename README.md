# Fortivex CloudPulse AI — Autonomous AI Sales Voice Agent

An end-to-end, real-time multilingual **Autonomous AI Sales Voice Agent** built for **Fortivex AI Private Limited (Round 3 Project Challenge)**.

## Implemented Features
- **Real-Time Voice Interaction (STT & TTS)**: Browser Web Speech API (`SpeechRecognition` + `SpeechSynthesis`) for zero-latency voice conversations, plus `/api/stt` (OpenAI Whisper) and `/api/tts` (OpenAI TTS) server endpoints.
- **Multilingual & Contextual Memory**: Persistent per-session conversation history in SQLite (`sales_agent.db`) with automatic language detection and responses across English, Hindi, Telugu, Tamil, and Spanish.
- **RAG-Based Knowledge Retrieval**: Pure-Python BM25 retriever (`app/rag.py`) indexing `knowledge_base/*.md` (product overview, pricing plans, security/DPDP compliance, case studies) with scored source citations on every turn.
- **Intelligent Lead Qualification & Sentiment Analysis**: Real-time BANT scoring (Budget, Authority, Need, Timeline, 0–100) + Sentiment (`Positive`, `Neutral`, `Skeptical`, `Frustrated`) + Intent (`Discovery`, `Pricing`, `Objection`, `Ready to Book`).
- **Automated Appointment Booking & CRM Sync**: Detects booking intent and persists confirmed demo slots into SQLite CRM (`leads` and `appointments` tables).
- **Call Summarisation & Business Insights**: One-click `/api/sessions/{session_id}/summary` generating executive summary, objection log, and next best action.
- **Zero-Failure Demo Mode**: Automatically uses **Gemini** (`GEMINI_API_KEY`) or **OpenAI** (`OPENAI_API_KEY`) when configured, and seamlessly falls back to a deterministic RAG + BANT + multilingual engine (`demo` mode) if offline or without API keys.

## Quick Start (Local Demo)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: add your GEMINI_API_KEY or OPENAI_API_KEY in .env
cp .env.example .env

# Run the server
uvicorn app.main:app --reload --port 8000
```
Open `http://localhost:8000` in Chrome/Edge for full microphone STT + TTS voice interaction.

## Run Automated Tests
```bash
python3 -m unittest discover -s tests -v
```

## System Architecture
See `architecture_diagram.png` and `ONE_PAGE_REPORT.md` for full MVP and 100,000-user production scaling details.
