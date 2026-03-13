"""Backward-compatible shim to canonical src.unipath.rag package."""

from src.unipath.rag import (  # noqa: F401
    EmbeddingModel,
    FAISSVectorStore,
    PDFProcessor,
    RAGPipeline,
    RAGRetriever,
    RAGSystemBuilder,
    SemanticChunker,
)

__all__ = [
    "PDFProcessor",
    "SemanticChunker",
    "EmbeddingModel",
    "FAISSVectorStore",
    "RAGRetriever",
    "RAGPipeline",
    "RAGSystemBuilder",
]
