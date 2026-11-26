from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import Settings
from app.crm import CRMStore
from app.models import (
    AppointmentBooking,
    BANTQualification,
    CallSummaryResponse,
    ChatRequest,
    ChatResponse,
    RAGCitation,
    SentimentAnalysis,
    from_dict,
    from_json,
    to_dict,
    to_json,
)
from app.rag import KnowledgeBaseRAG

SYSTEM_PROMPT = """You are Aria, an autonomous AI Sales Voice Agent for Fortivex CloudPulse AI.
Your goals:
1. Answer prospect questions accurately using ONLY the provided RAG Knowledge Base context.
2. Communicate concisely (2-4 spoken-friendly sentences) in the prospect's language (English, Hindi, Telugu, Tamil, Spanish, etc.).
3. Qualify the lead across BANT (Budget, Authority, Need, Timeline) and detect Sentiment + Intent.
4. When the prospect wants to schedule a demo and provides at least a name/email or asks to book a slot, populate `booking_request` to book the appointment in CRM.

Return ONLY valid JSON matching this exact schema:
{
  "reply": "string (natural, concise spoken response)",
  "detected_language": "en | hi | te | ta | es | other",
  "bant": {
    "budget": "string",
    "authority": "string",
    "need": "string",
    "timeline": "string",
    "score": 0-100,
    "tier": "Cold | Warm | Hot"
  },
  "sentiment": {
    "label": "Positive | Neutral | Skeptical | Frustrated",
    "confidence": 0.0-1.0,
    "intent": "Discovery | Pricing | Objection | Ready to Book | Support"
  },
  "booking_request": null OR {
    "lead_name": "string",
    "lead_email": "string",
    "company": "string",
    "slot_iso": "YYYY-MM-DD HH:MM IST",
    "notes": "string"
  }
}
"""


class SalesVoiceAgent:
    def __init__(self, settings: Settings, rag: KnowledgeBaseRAG, crm: CRMStore) -> None:
        self.settings = settings
        self.rag = rag
        self.crm = crm

    @staticmethod
    def _detect_language(text: str, requested: str) -> str:
        if requested and requested != "auto":
            return requested
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        if re.search(r"[\u0C00-\u0C7F]", text):
            return "te"
        if re.search(r"[\u0B80-\u0BFF]", text):
            return "ta"
        spanish_markers = ("hola", "precio", "demo", "gracias", "quiero", "buenos")
        if any(w in text.lower().split() for w in spanish_markers):
            return "es"
        hindi_roman = ("namaste", "kaise", "kitna", "paisa", "chahiye", "haan", "theek")
        if any(w in text.lower().split() for w in hindi_roman):
            return "hi"
        return "en"

    async def handle_turn(self, req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        session = self.crm.ensure_session(
            req.session_id,
            prospect_name=req.prospect_name,
            prospect_email=req.prospect_email,
            company=req.company,
        )
        history = self.crm.get_history(req.session_id, limit=10)
        citations = self.rag.search(req.message, top_k=3)
        lang = self._detect_language(req.message, req.language)

        prev_bant = from_json(BANTQualification, session["bant_json"])
        provider = self.settings.active_provider
        parsed: Optional[Dict[str, Any]] = None
        provider_used = provider

        if provider == "gemini":
            parsed = await self._call_gemini(req.message, history, citations, prev_bant, lang)
        elif provider == "openai":
            parsed = await self._call_openai(req.message, history, citations, prev_bant, lang)

        if parsed is None:
            provider_used = "demo"
            parsed = self._demo_reasoner(req, history, citations, prev_bant, lang)

        bant = from_dict(BANTQualification, parsed.get("bant", to_dict(prev_bant)))
        sentiment = from_dict(SentimentAnalysis, parsed.get("sentiment", {}))
        detected_lang = parsed.get("detected_language", lang)
        reply = parsed.get("reply", "I'd be happy to help with Fortivex CloudPulse AI.")

        booked: Optional[AppointmentBooking] = None
        booking_req = parsed.get("booking_request")
        if isinstance(booking_req, dict) and booking_req.get("slot_iso"):
            booked = self.crm.book_appointment(
                session_id=req.session_id,
                lead_name=booking_req.get("lead_name") or session["prospect_name"] or "Prospect",
                lead_email=booking_req.get("lead_email") or session["prospect_email"] or "prospect@company.com",
                company=booking_req.get("company") or session["company"] or "Evaluating Org",
                slot_iso=booking_req["slot_iso"],
                notes=booking_req.get("notes", "Booked via AI Voice Agent"),
            )

        self.crm.add_message(req.session_id, "user", req.message)
        self.crm.add_message(req.session_id, "assistant", reply)
        self.crm.update_intelligence(req.session_id, detected_lang, bant, sentiment)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return ChatResponse(
            session_id=req.session_id,
            reply=reply,
            detected_language=detected_lang,
            provider_used=provider_used,
            latency_ms=latency_ms,
            qualification=bant,
            sentiment=sentiment,
            citations=citations,
            booked_appointment=booked,
        )

    def _build_user_prompt(
        self,
        message: str,
        history: List[Dict[str, str]],
        citations: List[RAGCitation],
        prev_bant: BANTQualification,
        lang: str,
    ) -> str:
        kb_text = "\n".join(
            f"[{c.source} | {c.heading}]: {c.snippet}" for c in citations
        ) or "No specific KB match; answer from general Fortivex CloudPulse AI overview."
        hist_text = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in history[-6:])
        return (
            f"Target Language: {lang}\n"
            f"Previous BANT State: {to_json(prev_bant)}\n"
            f"RAG Knowledge Base Context:\n{kb_text}\n\n"
            f"Recent Conversation:\n{hist_text}\n\n"
            f"Latest Prospect Utterance: {message}"
        )

    async def _call_gemini(
        self,
        message: str,
        history: List[Dict[str, str]],
        citations: List[RAGCitation],
        prev_bant: BANTQualification,
        lang: str,
    ) -> Optional[Dict[str, Any]]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent?key={self.settings.gemini_api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": [{"text": self._build_user_prompt(message, history, citations, prev_bant, lang)}]}
            ],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except Exception:
            return None

    async def _call_openai(
        self,
        message: str,
        history: List[Dict[str, str]],
        citations: List[RAGCitation],
        prev_bant: BANTQualification,
        lang: str,
    ) -> Optional[Dict[str, Any]]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(message, history, citations, prev_bant, lang)},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return json.loads(text)
        except Exception:
            return None

    def _demo_reasoner(
        self,
        req: ChatRequest,
        history: List[Dict[str, str]],
        citations: List[RAGCitation],
        prev_bant: BANTQualification,
        lang: str,
    ) -> Dict[str, Any]:
        msg = req.message.lower()
        budget = prev_bant.budget
        authority = prev_bant.authority
        need = prev_bant.need
        timeline = prev_bant.timeline
        score = prev_bant.score

        if any(k in msg for k in ("budget", "price", "cost", "plan", "dollar", "inr", "₹", "$", "expensive")):
            budget = "Evaluating Starter ($199/mo) / Growth ($599/mo)"
            score = max(score, 40)
        if any(k in msg for k in ("ceo", "cto", "vp", "head", "founder", "manager", "decision", "team")):
            authority = "Decision-Maker / Stakeholder"
            score = min(100, score + 20)
        if any(k in msg for k in ("lead", "call", "sales", "crm", "missed", "automate", "support", "booking", "voice")):
            need = "Automated inbound voice qualification & CRM sync"
            score = min(100, max(score, 45) + 15)
        if any(k in msg for k in ("week", "month", "today", "tomorrow", "asap", "q1", "q2", "soon", "urgent")):
            timeline = "Immediate / Near-term (2-4 weeks)"
            score = min(100, score + 20)

        sentiment_label = "Neutral"
        intent = "Discovery"
        if any(k in msg for k in ("expensive", "too much", "not sure", "concern", "doubt", "worried")):
            sentiment_label = "Skeptical"
            intent = "Objection"
        elif any(k in msg for k in ("frustrated", "bad", "slow", "waste", "terrible")):
            sentiment_label = "Frustrated"
            intent = "Objection"
        elif any(k in msg for k in ("great", "awesome", "love", "interested", "sounds good", "book", "schedule", "demo")):
            sentiment_label = "Positive"
            intent = "Ready to Book" if any(k in msg for k in ("book", "schedule", "demo", "slot", "meeting")) else "Discovery"
        elif any(k in msg for k in ("price", "pricing", "plan", "cost", "how much")):
            intent = "Pricing"

        tier = "Hot" if score >= 70 else ("Warm" if score >= 40 else "Cold")

        booking_request = None
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", req.message)
        wants_book = any(k in msg for k in ("book", "schedule", "demo", "appointment", "slot", "tomorrow"))
        if wants_book and (email_match or req.prospect_email or len(history) >= 2):
            email = email_match.group(0) if email_match else (req.prospect_email or "lead@prospect.io")
            booking_request = {
                "lead_name": req.prospect_name or "Valued Prospect",
                "lead_email": email,
                "company": req.company or "Prospect Corp",
                "slot_iso": "2026-11-27 11:00 IST",
                "notes": f"Auto-booked in {lang.upper()} session; BANT score {score}/100",
            }
            score = max(score, 80)
            tier = "Hot"

        kb_answer = citations[0].snippet if citations else (
            "Fortivex CloudPulse AI provides 24/7 multilingual voice qualification under 400ms latency with native CRM and calendar sync."
        )

        if booking_request:
            if lang == "hi":
                reply = (
                    f"बहुत बढ़िया! मैंने आपका डेमो कल 27 नवंबर को सुबह 11:00 बजे IST के लिए "
                    f"{booking_request['lead_email']} पर बुक कर दिया है और CRM में अपडेट कर दिया है।"
                )
            elif lang == "es":
                reply = (
                    f"¡Perfecto! He reservado tu demo para mañana 27 de noviembre a las 11:00 IST "
                    f"para {booking_request['lead_email']} y lo sincronicé con el CRM."
                )
            else:
                reply = (
                    f"Done! I've booked your live CloudPulse AI demo for tomorrow, 27 Nov at 11:00 AM IST "
                    f"({booking_request['lead_email']}) and synced your qualification profile to the CRM."
                )
        elif intent == "Pricing":
            if lang == "hi":
                reply = (
                    "हमारा Starter प्लान $199/माह (1,000 मिनट) से शुरू होता है और Growth प्लान $599/माह "
                    "(5,000 मिनट, 25+ भाषाएं और CRM सिंक) में आता है। क्या आप 14-दिन का फ्री पायलट देखना चाहेंगे?"
                )
            else:
                reply = (
                    "Our Starter plan is $199/month (1,000 minutes) and Growth is $599/month "
                    "(5,000 minutes, 25+ languages, custom RAG, and CRM sync), plus a 14-day free pilot. "
                    "Would you like me to schedule a quick 15-minute walkthrough?"
                )
        elif intent == "Objection":
            reply = (
                "That's a fair concern. Based on our FinServe India deployment, teams cut cost-per-qualified-lead by 58% "
                "and see 3.1x more booked demos—plus we include a 14-day free pilot with SOC 2 Type II & DPDP compliance. "
                "What monthly call volume are you currently handling?"
            )
        else:
            if lang == "hi":
                reply = (
                    "नमस्ते! Fortivex CloudPulse AI में आपका स्वागत है। हम 25+ भाषाओं में 24/7 AI सेल्स कॉल, "
                    "BANT क्वालिफिकेशन और ऑटोमैटिक CRM बुकिंग देते हैं। आपकी टीम अभी महीने में कितनी लीड्स हैंडल करती है?"
                )
            elif lang == "es":
                reply = (
                    "¡Hola! Bienvenido a Fortivex CloudPulse AI. Automatizamos llamadas de ventas en más de 25 idiomas "
                    "con calificación BANT y sincronización CRM. ¿Qué volumen de leads manejan al mes?"
                )
            else:
                reply = (
                    f"Great question! {kb_answer} "
                    "To tailor this for your team, roughly how many inbound leads are you handling each month?"
                )

        return {
            "reply": reply,
            "detected_language": lang,
            "bant": {
                "budget": budget,
                "authority": authority,
                "need": need,
                "timeline": timeline,
                "score": score,
                "tier": tier,
            },
            "sentiment": {
                "label": sentiment_label,
                "confidence": 0.87,
                "intent": intent,
            },
            "booking_request": booking_request,
        }

    def summarize_call(self, session_id: str) -> CallSummaryResponse:
        session = self.crm.ensure_session(session_id)
        history = self.crm.get_history(session_id, limit=50)
        bant = from_json(BANTQualification, session["bant_json"])
        sent = from_json(SentimentAnalysis, session["sentiment_json"])

        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        objections = []
        for msg in user_msgs:
            low = msg.lower()
            if any(w in low for w in ("expensive", "price", "cost", "security", "privacy", "concern")):
                objections.append(msg[:120])

        if not objections:
            objections = ["No major blockers raised during discovery."]

        next_action = (
            "Send calendar invite & technical sandbox credentials before tomorrow's AE demo."
            if bant.score >= 70
            else "Follow up with ROI case study (FinServe India) and 14-day free pilot offer."
        )

        summary_text = (
            f"Prospect {session['prospect_name']} ({session['company'] or 'Unspecified Org'}) "
            f"completed {len(user_msgs)} turns in {session['language'].upper()}. "
            f"Need: {bant.need}. Budget: {bant.budget}. Timeline: {bant.timeline}. "
            f"Lead scored {bant.score}/100 ({bant.tier}) with {sent.label} overall sentiment."
        )

        payload = {
            "session_id": session_id,
            "executive_summary": summary_text,
            "qualification_score": bant.score,
            "qualification_tier": bant.tier,
            "sentiment_overall": sent.label,
            "key_objections": objections[:3],
            "next_best_action": next_action,
            "crm_synced": True,
            "turn_count": len(user_msgs),
        }
        self.crm.save_summary(session_id, payload)
        return CallSummaryResponse(**payload)
