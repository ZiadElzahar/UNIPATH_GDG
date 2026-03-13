"""Backward-compatible shim to src.unipath.rag.llm_clients."""

from src.unipath.rag.llm_clients import (
    BaseLLMClient,
    GoogleGeminiClient,
    HuggingFaceClient,
    OllamaClient,
    OpenAIClient,
    get_llm_client,
)

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "GoogleGeminiClient",
    "OllamaClient",
    "HuggingFaceClient",
    "get_llm_client",
]
