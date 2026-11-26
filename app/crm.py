from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.models import AppointmentBooking, BANTQualification, SentimentAnalysis, to_json


class CRMStore:
    """SQLite-backed session memory, lead qualification CRM, and appointment scheduler."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    prospect_name TEXT DEFAULT 'Prospect',
                    prospect_email TEXT DEFAULT '',
                    company TEXT DEFAULT '',
                    language TEXT DEFAULT 'en',
                    bant_json TEXT NOT NULL,
                    sentiment_json TEXT NOT NULL,
                    summary_json TEXT DEFAULT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    lead_name TEXT NOT NULL,
                    lead_email TEXT NOT NULL,
                    company TEXT NOT NULL,
                    slot_iso TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    status TEXT DEFAULT 'Confirmed',
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def ensure_session(
        self,
        session_id: str,
        prospect_name: Optional[str] = None,
        prospect_email: Optional[str] = None,
        company: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            now = self._now_iso()
            if row is None:
                default_bant = to_json(BANTQualification())
                default_sent = to_json(SentimentAnalysis())
                conn.execute(
                    """
                    INSERT INTO sessions
                    (session_id, prospect_name, prospect_email, company, language, bant_json, sentiment_json, updated_at)
                    VALUES (?, ?, ?, ?, 'en', ?, ?, ?)
                    """,
                    (
                        session_id,
                        prospect_name or "Prospect",
                        prospect_email or "",
                        company or "",
                        default_bant,
                        default_sent,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
            elif any([prospect_name, prospect_email, company]):
                conn.execute(
                    """
                    UPDATE sessions
                    SET prospect_name = COALESCE(NULLIF(?, ''), prospect_name),
                        prospect_email = COALESCE(NULLIF(?, ''), prospect_email),
                        company = COALESCE(NULLIF(?, ''), company),
                        updated_at = ?
                    WHERE session_id = ?
                    """,
                    (prospect_name or "", prospect_email or "", company or "", now, session_id),
                )
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
            return dict(row)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, self._now_iso()),
            )

    def get_history(self, session_id: str, limit: int = 12) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def update_intelligence(
        self,
        session_id: str,
        language: str,
        bant: BANTQualification,
        sentiment: SentimentAnalysis,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET language = ?, bant_json = ?, sentiment_json = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (language, to_json(bant), to_json(sentiment), self._now_iso(), session_id),
            )

    def book_appointment(
        self,
        session_id: str,
        lead_name: str,
        lead_email: str,
        company: str,
        slot_iso: str,
        notes: str = "",
    ) -> AppointmentBooking:
        appt_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO appointments
                (appointment_id, session_id, lead_name, lead_email, company, slot_iso, notes, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'Confirmed', ?)
                """,
                (appt_id, session_id, lead_name, lead_email, company, slot_iso, notes, self._now_iso()),
            )
            conn.execute(
                """
                UPDATE sessions
                SET prospect_name = ?, prospect_email = ?, company = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (lead_name, lead_email, company, self._now_iso(), session_id),
            )
        return AppointmentBooking(
            appointment_id=appt_id,
            lead_name=lead_name,
            lead_email=lead_email,
            company=company,
            slot_iso=slot_iso,
            notes=notes,
            status="Confirmed",
        )

    def save_summary(self, session_id: str, summary: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET summary_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(summary), self._now_iso(), session_id),
            )

    def list_leads(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 50"
            ).fetchall()
        leads: List[Dict[str, Any]] = []
        for r in rows:
            bant = json.loads(r["bant_json"])
            sent = json.loads(r["sentiment_json"])
            leads.append(
                {
                    "session_id": r["session_id"],
                    "prospect_name": r["prospect_name"],
                    "prospect_email": r["prospect_email"],
                    "company": r["company"],
                    "language": r["language"],
                    "bant": bant,
                    "sentiment": sent,
                    "has_summary": r["summary_json"] is not None,
                    "updated_at": r["updated_at"],
                }
            )
        return leads

    def list_appointments(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM appointments ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        return [dict(r) for r in rows]
