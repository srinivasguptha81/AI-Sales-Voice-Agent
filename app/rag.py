from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.models import RAGCitation

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "what", "which", "who", "how", "when",
    "where", "why", "this", "that", "these", "those", "it", "its", "you", "your",
    "we", "our", "i", "me", "my",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


@dataclass
class KBChunk:
    chunk_id: str
    source: str
    heading: str
    content: str
    tokens: List[str]
    tf: Counter


class KnowledgeBaseRAG:
    """Pure-Python BM25 retriever over markdown files in knowledge_base/."""

    def __init__(self, kb_dir: Path) -> None:
        self.kb_dir = kb_dir
        self.chunks: List[KBChunk] = []
        self.idf: dict[str, float] = {}
        self.avg_len: float = 1.0
        self.reload()

    def reload(self) -> None:
        self.chunks.clear()
        if not self.kb_dir.exists():
            return

        for md_file in sorted(self.kb_dir.glob("*.md")):
            raw = md_file.read_text(encoding="utf-8")
            sections = re.split(r"(?m)^##\s+", raw)
            doc_title = md_file.stem.replace("_", " ").title()
            for idx, sec in enumerate(sections):
                sec = sec.strip()
                if not sec:
                    continue
                lines = sec.splitlines()
                heading = lines[0].lstrip("# ").strip() if lines else doc_title
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else sec
                full_text = f"{heading}\n{body}".strip()
                tokens = _tokenize(full_text)
                if not tokens:
                    continue
                self.chunks.append(
                    KBChunk(
                        chunk_id=f"{md_file.stem}#{idx}",
                        source=md_file.name,
                        heading=heading,
                        content=full_text,
                        tokens=tokens,
                        tf=Counter(tokens),
                    )
                )

        n_docs = len(self.chunks)
        if n_docs == 0:
            self.avg_len = 1.0
            self.idf = {}
            return

        self.avg_len = sum(len(c.tokens) for c in self.chunks) / n_docs
        df: Counter = Counter()
        for c in self.chunks:
            for term in set(c.tokens):
                df[term] += 1

        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 3) -> List[RAGCitation]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        k1 = 1.5
        b = 0.75
        scored: List[tuple[float, KBChunk]] = []

        for chunk in self.chunks:
            score = 0.0
            doc_len = len(chunk.tokens)
            for term in q_tokens:
                if term not in chunk.tf:
                    continue
                tf = chunk.tf[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + k1 * (1.0 - b + b * (doc_len / self.avg_len))
                score += idf * ((tf * (k1 + 1.0)) / denom)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: List[RAGCitation] = []
        for score, chunk in scored[:top_k]:
            snippet = chunk.content.replace("\n", " ")[:240]
            if len(chunk.content) > 240:
                snippet += "..."
            results.append(
                RAGCitation(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    heading=chunk.heading,
                    score=round(score, 3),
                    snippet=snippet,
                )
            )
        return results
