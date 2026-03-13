"""Backward-compatible shim to src.unipath.rag.retriever."""

from src.unipath.rag.retriever import RAGRetriever, RetrievalResult

__all__ = ["RetrievalResult", "RAGRetriever"]
