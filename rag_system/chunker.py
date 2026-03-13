"""Backward-compatible shim to src.unipath.rag.chunker."""

from src.unipath.rag.chunker import Chunk, SemanticChunker, create_rag_dataset

__all__ = ["Chunk", "SemanticChunker", "create_rag_dataset"]
