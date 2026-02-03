"""
Semantic Chunker Module
Handles intelligent text chunking with overlap for better context preservation.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    start_index: int = 0
    end_index: int = 0
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "start_index": self.start_index,
            "end_index": self.end_index
        }


class SemanticChunker:
    """
    Splits text into semantically meaningful chunks with configurable overlap.
    Optimized for Arabic academic documents.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
        separator_pattern: Optional[str] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
            min_chunk_size: Minimum words for a valid chunk
            separator_pattern: Regex pattern for splitting (optional)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Default separators for Arabic academic text
        self.separator_pattern = separator_pattern or r'(?:مادة\s*[:\(]?\s*\d+|[.!؟\n]{2,})'
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        chunks = []
        
        # Clean the text first
        text = self._preprocess_text(text)
        
        # Try semantic splitting first (by articles/sections)
        semantic_chunks = self._split_by_semantics(text)
        
        if semantic_chunks:
            # Process each semantic section
            for section in semantic_chunks:
                section_chunks = self._split_by_size(
                    section['text'],
                    {**metadata, **section.get('metadata', {})}
                )
                chunks.extend(section_chunks)
        else:
            # Fall back to size-based splitting
            chunks = self._split_by_size(text, metadata)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for chunking.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve paragraph breaks
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        return text.strip()
    
    def _split_by_semantics(self, text: str) -> List[Dict]:
        """
        Split text by semantic boundaries (articles, sections).
        
        Args:
            text: Input text
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # Find article boundaries
        article_pattern = r'مادة\s*[:\(]?\s*(\d+)\s*[:\)]?'
        matches = list(re.finditer(article_pattern, text))
        
        if len(matches) < 2:
            return []
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[start:end].strip()
            article_num = match.group(1)
            
            if len(section_text.split()) >= self.min_chunk_size:
                sections.append({
                    'text': section_text,
                    'metadata': {
                        'article_number': int(article_num),
                        'section_type': 'article'
                    }
                })
        
        return sections
    
    def _split_by_size(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Input text
            metadata: Metadata to include
            
        Returns:
            List of Chunk objects
        """
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            # Text is smaller than chunk size
            if len(words) >= self.min_chunk_size:
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    text=text,
                    metadata=metadata,
                    start_index=0,
                    end_index=len(text)
                )
                chunks.append(chunk)
            return chunks
        
        # Split into overlapping chunks
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_words) >= self.min_chunk_size:
                chunk = Chunk(
                    chunk_id=f"{metadata.get('article_number', 'doc')}_{chunk_num}",
                    text=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_number': chunk_num,
                        'word_count': len(chunk_words)
                    },
                    start_index=start_idx,
                    end_index=end_idx
                )
                chunks.append(chunk)
                chunk_num += 1
            
            # Move start with overlap
            start_idx = end_idx - self.chunk_overlap
            
            # Avoid infinite loop
            if start_idx >= len(words) - self.min_chunk_size:
                break
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries with 'text' and optional 'metadata'
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks


def create_rag_dataset(
    text: str,
    source_name: str,
    chunk_size: int = 200,
    chunk_overlap: int = 50
) -> List[Dict]:
    """
    Create a RAG-ready dataset from text.
    
    Args:
        text: Input text
        source_name: Name of the source document
        chunk_size: Target words per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries ready for embedding
    """
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = chunker.chunk_text(text, {'source': source_name})
    
    return [
        {
            'chunk_id': i + 1,
            'text': chunk.text,
            'metadata': chunk.metadata
        }
        for i, chunk in enumerate(chunks)
    ]
