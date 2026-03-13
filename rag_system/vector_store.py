"""Backward-compatible shim to src.unipath.rag.vector_store."""

from src.unipath.rag.vector_store import FAISSVectorStore, SearchResult

__all__ = ["SearchResult", "FAISSVectorStore"]
