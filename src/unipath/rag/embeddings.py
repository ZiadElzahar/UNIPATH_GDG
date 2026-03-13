"""
Embedding Model Module
Handles text embedding generation using sentence transformers.
"""

import os
from typing import List, Optional, Union

import numpy as np


class EmbeddingModel:
    ARABIC_MODELS = {
        "multilingual-minilm": "paraphrase-multilingual-MiniLM-L12-v2",
        "multilingual-mpnet": "paraphrase-multilingual-mpnet-base-v2",
        "arabic-bert": "aubmindlab/bert-base-arabertv2",
        "labse": "sentence-transformers/LaBSE",
    }

    def __init__(self, model_name: str = "multilingual-minilm", device: Optional[str] = None, cache_dir: Optional[str] = None):
        self.model_name = self.ARABIC_MODELS.get(model_name, model_name)
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.embedding_dim = None

    def load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers") from exc

        kwargs = {}
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir
        if self.device:
            kwargs["device"] = self.device

        self.model = SentenceTransformer(self.model_name, **kwargs)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress: bool = True, normalize: bool = True) -> np.ndarray:
        if self.model is None:
            self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        texts = [self._preprocess(text) for text in texts]
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, normalize_embeddings=normalize)
        return np.array(embeddings).astype("float32")

    def _preprocess(self, text: str) -> str:
        text = " ".join(text.split())
        if len(text) > 5000:
            text = text[:5000]
        return text

    def get_embedding_dimension(self) -> int:
        if self.embedding_dim is None:
            self.load_model()
        return self.embedding_dim

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        return 0.0

    def batch_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        return np.dot(query_norm, doc_norms.T).flatten()


class EmbeddingCache:
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, identifier: str) -> str:
        return os.path.join(self.cache_dir, f"{identifier}_embeddings.npy")

    def save(self, identifier: str, embeddings: np.ndarray) -> None:
        np.save(self.get_cache_path(identifier), embeddings)

    def load(self, identifier: str) -> Optional[np.ndarray]:
        cache_path = self.get_cache_path(identifier)
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None

    def exists(self, identifier: str) -> bool:
        return os.path.exists(self.get_cache_path(identifier))
