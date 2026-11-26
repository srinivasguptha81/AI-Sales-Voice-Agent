from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
STATIC_DIR = BASE_DIR / "static"


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    openai_api_key: str
    openai_model: str
    database_path: str

    @property
    def active_provider(self) -> str:
        mode = self.llm_provider.lower().strip()
        if mode == "gemini" and self.gemini_api_key:
            return "gemini"
        if mode == "openai" and self.openai_api_key:
            return "openai"
        if mode == "demo":
            return "demo"
        # 'auto' mode
        if self.gemini_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        return "demo"


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "auto"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        database_path=os.getenv("DATABASE_PATH", str(BASE_DIR / "sales_agent.db")),
    )
