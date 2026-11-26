from __future__ import annotations

import os
import tempfile
import unittest

os.environ["LLM_PROVIDER"] = "demo"
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_PATH"] = _TMP_DB.name

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


class SalesVoiceAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_and_ui(self) -> None:
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertGreater(data["kb_chunks"], 0)

        ui = self.client.get("/")
        self.assertEqual(ui.status_code, 200)
        self.assertIn("Fortivex CloudPulse AI", ui.text)

    def test_end_to_end_sales_workflow(self) -> None:
        sid = "test-session-e2e"

        # 1. Discovery + RAG pricing
        r1 = self.client.post(
            "/api/chat",
            json={
                "session_id": sid,
                "message": "What is the pricing for the Growth plan and CRM integration?",
                "prospect_name": "Riya Sharma",
                "prospect_email": "riya@acme.in",
                "company": "Acme Fintech",
            },
        )
        self.assertEqual(r1.status_code, 200)
        d1 = r1.json()
        self.assertGreater(len(d1["citations"]), 0)
        self.assertGreaterEqual(d1["qualification"]["score"], 40)

        # 2. Multilingual Hindi turn
        r2 = self.client.post(
            "/api/chat",
            json={
                "session_id": sid,
                "message": "हमारा बजट तैयार है, क्या आप हिंदी में बता सकते हैं?",
            },
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["detected_language"], "hi")

        # 3. Appointment booking
        r3 = self.client.post(
            "/api/chat",
            json={
                "session_id": sid,
                "message": "Great, please book a demo for tomorrow at 11 AM for riya@acme.in",
            },
        )
        self.assertEqual(r3.status_code, 200)
        d3 = r3.json()
        self.assertIsNotNone(d3["booked_appointment"])
        self.assertEqual(d3["booked_appointment"]["lead_email"], "riya@acme.in")

        # 4. Call Summary & Business Insights
        r4 = self.client.post(f"/api/sessions/{sid}/summary")
        self.assertEqual(r4.status_code, 200)
        s = r4.json()
        self.assertTrue(s["crm_synced"])
        self.assertEqual(s["turn_count"], 3)

        # 5. CRM Appointments check
        appts = self.client.get("/api/crm/appointments").json()["appointments"]
        self.assertTrue(any(a["lead_email"] == "riya@acme.in" for a in appts))


if __name__ == "__main__":
    unittest.main()
