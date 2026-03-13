"""
Semantic Chunker Module
Handles intelligent text chunking with overlap for better context preservation.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    chunk_id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    start_index: int = 0
    end_index: int = 0

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
        separator_pattern: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separator_pattern = separator_pattern or r"(?:مادة\s*[:\(]?\s*\d+|[.!؟\n]{2,})"

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        chunks = []

        text = self._preprocess_text(text)
        semantic_chunks = self._split_by_semantics(text)

        if semantic_chunks:
            for section in semantic_chunks:
                section_chunks = self._split_by_size(section["text"], {**metadata, **section.get("metadata", {})})
                chunks.extend(section_chunks)
        else:
            chunks = self._split_by_size(text, metadata)

        return chunks

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text.strip()

    def _split_by_semantics(self, text: str) -> List[Dict]:
        sections = []
        article_pattern = r"مادة\s*[:\(]?\s*(\d+)\s*[:\)]?"
        matches = list(re.finditer(article_pattern, text))

        if len(matches) < 2:
            return []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            section_text = text[start:end].strip()
            article_num = match.group(1)

            if len(section_text.split()) >= self.min_chunk_size:
                sections.append(
                    {
                        "text": section_text,
                        "metadata": {"article_number": int(article_num), "section_type": "article"},
                    }
                )

        return sections

    def _split_by_size(self, text: str, metadata: Dict) -> List[Chunk]:
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            if len(words) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4())[:8],
                        text=text,
                        metadata=metadata,
                        start_index=0,
                        end_index=len(text),
                    )
                )
            return chunks

        start_idx = 0
        chunk_num = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            if len(chunk_words) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id=f"{metadata.get('article_number', 'doc')}_{chunk_num}",
                        text=chunk_text,
                        metadata={**metadata, "chunk_number": chunk_num, "word_count": len(chunk_words)},
                        start_index=start_idx,
                        end_index=end_idx,
                    )
                )
                chunk_num += 1

            start_idx = end_idx - self.chunk_overlap
            if start_idx >= len(words) - self.min_chunk_size:
                break

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc.get("text", ""), doc.get("metadata", {}))
            all_chunks.extend(chunks)
        return all_chunks


def create_rag_dataset(text: str, source_name: str, chunk_size: int = 200, chunk_overlap: int = 50) -> List[Dict]:
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_text(text, {"source": source_name})

    return [{"chunk_id": i + 1, "text": chunk.text, "metadata": chunk.metadata} for i, chunk in enumerate(chunks)]
