"""Backward-compatible shim to src.unipath.rag.pdf_processor."""

from src.unipath.rag.pdf_processor import PDFProcessor, PageContent, extract_pdf_to_json

__all__ = ["PageContent", "PDFProcessor", "extract_pdf_to_json"]
