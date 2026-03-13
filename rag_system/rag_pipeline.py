"""Backward-compatible shim to src.unipath.rag.rag_pipeline."""

from src.unipath.rag.rag_pipeline import RAGPipeline, RAGResponse, RAGSystemBuilder

__all__ = ["RAGResponse", "RAGPipeline", "RAGSystemBuilder"]
