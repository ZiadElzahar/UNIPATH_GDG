"""Retriever Module."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RetrievalResult:
    text: str
    score: float
    metadata: Dict
    chunk_id: str
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }


class RAGRetriever:
    def __init__(self, vector_store, embedding_model, reranker=None):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.arabic_stop_words = {
            "في",
            "من",
            "على",
            "إلى",
            "عن",
            "مع",
            "هذا",
            "هذه",
            "التي",
            "الذي",
            "ما",
            "هل",
            "كيف",
            "أين",
            "متى",
            "لماذا",
            "و",
            "أو",
            "ثم",
            "لكن",
            "بل",
            "حتى",
            "إذا",
            "إن",
            "أن",
            "كان",
            "يكون",
            "تكون",
            "كانت",
            "كانوا",
        }

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.3, rerank: bool = True, expand_query: bool = True) -> List[RetrievalResult]:
        processed_query = self._preprocess_query(query)
        expanded_queries = self._expand_query(processed_query) if expand_query else [processed_query]

        all_results = []
        seen_chunks = set()

        for q in expanded_queries:
            query_embedding = self.embedding_model.encode(q, show_progress=False)
            results = self.vector_store.search(query_embedding, k=k * 2, threshold=threshold)
            for result in results:
                if result.chunk_id not in seen_chunks:
                    all_results.append(
                        RetrievalResult(
                            text=result.text,
                            score=result.score,
                            metadata=result.metadata,
                            chunk_id=result.chunk_id,
                        )
                    )
                    seen_chunks.add(result.chunk_id)

        all_results.sort(key=lambda x: x.score, reverse=True)
        if rerank and len(all_results) > 0:
            all_results = self._rerank(query, all_results[: k * 2])

        return all_results[:k]

    def _preprocess_query(self, query: str) -> str:
        query = " ".join(query.strip().split())
        query = query.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        return query

    def _expand_query(self, query: str) -> List[str]:
        queries = [query]
        expansions = {
            "شروط": ["متطلبات", "requirements"],
            "تخرج": ["graduation", "إنهاء الدراسة"],
            "مقرر": ["course", "مادة دراسية"],
            "ساعات": ["credits", "وحدات"],
            "معتمدة": ["credit hours", "معتمده"],
            "تسجيل": ["registration", "قيد"],
            "حذف": ["drop", "إسقاط"],
            "إضافة": ["add", "إلحاق"],
            "معدل": ["GPA", "تقدير تراكمي"],
            "فصل": ["semester", "ترم"],
            "رسوب": ["fail", "إخفاق"],
            "نجاح": ["pass", "اجتياز"],
        }

        for term, alternatives in expansions.items():
            if term in query.lower():
                for alt in alternatives:
                    new_query = query.replace(term, alt)
                    if new_query != query:
                        queries.append(new_query)

        return queries[:3]

    def _rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        return self._keyword_rerank(query, results)

    def _keyword_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        query_words = set(query.lower().split()) - self.arabic_stop_words
        for result in results:
            text_lower = result.text.lower()
            matches = sum(1 for word in query_words if word in text_lower)
            keyword_score = matches / max(len(query_words), 1)
            result.rerank_score = 0.7 * result.score + 0.3 * keyword_score
        results.sort(key=lambda x: x.rerank_score or x.score, reverse=True)
        return results

    def retrieve_by_article(self, article_number: int, k: int = 3) -> List[RetrievalResult]:
        results = []
        for doc in self.vector_store.documents:
            if doc and doc.get("metadata", {}).get("article_number") == article_number:
                results.append(RetrievalResult(text=doc["text"], score=1.0, metadata=doc["metadata"], chunk_id=doc["chunk_id"]))
        return results[:k]

    def get_context(self, query: str, k: int = 3, max_tokens: int = 2000) -> Tuple[str, List[Dict]]:
        results = self.retrieve(query, k=k)
        context_parts = []
        sources = []
        total_length = 0

        for i, result in enumerate(results):
            text_length = len(result.text.split())
            if total_length + text_length > max_tokens:
                break

            context_parts.append(f"[Document {i+1}]:\n{result.text}")
            sources.append(
                {
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "metadata": result.metadata,
                    "text": result.text,
                }
            )
            total_length += text_length

        return "\n\n".join(context_parts), sources
