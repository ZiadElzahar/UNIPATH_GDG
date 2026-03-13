"""Vector Store Module."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Dict
    chunk_id: str

    def to_dict(self) -> Dict:
        return {"text": self.text, "score": self.score, "metadata": self.metadata, "chunk_id": self.chunk_id}


class FAISSVectorStore:
    def __init__(self, embedding_dim: int, index_type: str = "flat", metric: str = "cosine"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.documents: List[Dict] = []
        self.id_to_idx: Dict[str, int] = {}
        self._initialize_index()

    def _initialize_index(self) -> None:
        import faiss

        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIDMap(self.index)

    def add(self, embeddings: np.ndarray, documents: List[Dict], ids: Optional[List[str]] = None) -> None:
        import faiss

        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        if ids is None:
            start_id = len(self.documents)
            ids = [str(i + start_id) for i in range(len(documents))]

        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)

        numeric_ids = np.array([len(self.documents) + i for i in range(len(documents))], dtype=np.int64)
        self.index.add_with_ids(embeddings.astype("float32"), numeric_ids)

        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            idx = len(self.documents)
            self.documents.append({"text": doc.get("text", ""), "metadata": doc.get("metadata", {}), "chunk_id": doc_id})
            self.id_to_idx[doc_id] = idx

    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> List[SearchResult]:
        import faiss

        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding.astype("float32"), min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            similarity = 1 / (1 + score) if self.metric == "l2" else float(score)
            if similarity < threshold:
                continue

            doc = self.documents[idx]
            results.append(SearchResult(text=doc["text"], score=similarity, metadata=doc["metadata"], chunk_id=doc["chunk_id"]))

        return results

    def save(self, directory: str) -> None:
        import faiss

        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": self.documents,
                    "id_to_idx": self.id_to_idx,
                    "embedding_dim": self.embedding_dim,
                    "metric": self.metric,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str) -> "FAISSVectorStore":
        import faiss

        with open(os.path.join(directory, "documents.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls(embedding_dim=data["embedding_dim"], metric=data["metric"])
        store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        store.documents = data["documents"]
        store.id_to_idx = data["id_to_idx"]
        return store

    def __len__(self) -> int:
        return len([d for d in self.documents if d is not None])
