from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


def to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def to_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return model.json()


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    return cls.parse_obj(data)


def from_json(cls: Type[T], raw: str) -> T:
    if hasattr(cls, "model_validate_json"):
        return cls.model_validate_json(raw)
    return cls.parse_obj(json.loads(raw))


class RAGCitation(BaseModel):
    chunk_id: str
    source: str
    heading: str
    score: float
    snippet: str


class BANTQualification(BaseModel):
    budget: str = "Unknown"
    authority: str = "Unknown"
    need: str = "Unknown"
    timeline: str = "Unknown"
    score: int = Field(default=15, ge=0, le=100)
    tier: str = "Cold"  # Cold (<40), Warm (40-69), Hot (70+)


class SentimentAnalysis(BaseModel):
    label: str = "Neutral"  # Positive | Neutral | Skeptical | Frustrated
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    intent: str = "Discovery"  # Discovery | Pricing | Objection | Ready to Book | Support


class AppointmentBooking(BaseModel):
    appointment_id: str
    lead_name: str
    lead_email: str
    company: str
    slot_iso: str
    notes: str = ""
    status: str = "Confirmed"


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    language: str = Field(default="auto", description="auto | en | hi | es | te | ta")
    prospect_name: Optional[str] = None
    prospect_email: Optional[str] = None
    company: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    detected_language: str
    provider_used: str
    latency_ms: int
    qualification: BANTQualification
    sentiment: SentimentAnalysis
    citations: List[RAGCitation] = Field(default_factory=list)
    booked_appointment: Optional[AppointmentBooking] = None


class CallSummaryResponse(BaseModel):
    session_id: str
    executive_summary: str
    qualification_score: int
    qualification_tier: str
    sentiment_overall: str
    key_objections: List[str]
    next_best_action: str
    crm_synced: bool
    turn_count: int


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    voice: str = "alloy"
