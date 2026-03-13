"""Backward-compatible shim to src.unipath.rag.embeddings."""

from src.unipath.rag.embeddings import EmbeddingCache, EmbeddingModel

__all__ = ["EmbeddingModel", "EmbeddingCache"]
