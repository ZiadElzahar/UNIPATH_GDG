"""
Vector Store Module
FAISS-based vector storage and similarity search.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Represents a search result with score and metadata."""
    text: str
    score: float
    metadata: Dict
    chunk_id: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id
        }


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    Supports multiple index types and persistence.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = 'flat',
        metric: str = 'cosine'
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.documents: List[Dict] = []
        self.id_to_idx: Dict[str, int] = {}
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize the FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required. Install with: "
                "pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        # Create base index based on metric
        if self.metric == 'cosine':
            # For cosine similarity, normalize vectors and use inner product
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric == 'l2':
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:  # inner product
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Wrap with ID map for document tracking
        self.index = faiss.IndexIDMap(self.index)
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add embeddings and documents to the index.
        
        Args:
            embeddings: NumPy array of embeddings (N x D)
            documents: List of document dictionaries with 'text' and 'metadata'
            ids: Optional list of document IDs
        """
        import faiss
        
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Generate IDs if not provided
        if ids is None:
            start_id = len(self.documents)
            ids = [str(i + start_id) for i in range(len(documents))]
        
        # Normalize embeddings for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        # Create numeric IDs for FAISS
        numeric_ids = np.array(
            [len(self.documents) + i for i in range(len(documents))],
            dtype=np.int64
        )
        
        # Add to index
        self.index.add_with_ids(embeddings.astype('float32'), numeric_ids)
        
        # Store document metadata
        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            idx = len(self.documents)
            self.documents.append({
                'text': doc.get('text', ''),
                'metadata': doc.get('metadata', {}),
                'chunk_id': doc_id
            })
            self.id_to_idx[doc_id] = idx
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        import faiss
        
        if self.index.ntotal == 0:
            return []
        
        # Ensure proper shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            min(k, self.index.ntotal)
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            
            # Convert L2 distance to similarity if needed
            if self.metric == 'l2':
                similarity = 1 / (1 + score)
            else:
                similarity = float(score)
            
            if similarity < threshold:
                continue
            
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc['text'],
                score=similarity,
                metadata=doc['metadata'],
                chunk_id=doc['chunk_id']
            ))
        
        return results
    
    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        filter_fn,
        k: int = 5,
        search_k: int = 50
    ) -> List[SearchResult]:
        """
        Search with a filter function applied to results.
        
        Args:
            query_embedding: Query embedding vector
            filter_fn: Function that takes a document dict and returns bool
            k: Number of results to return after filtering
            search_k: Number of candidates to search before filtering
            
        Returns:
            Filtered list of SearchResult objects
        """
        # Get more results than needed to account for filtering
        results = self.search(query_embedding, k=search_k)
        
        # Apply filter
        filtered = [r for r in results if filter_fn(r)]
        
        return filtered[:k]
    
    def delete(self, doc_ids: List[str]) -> int:
        """
        Delete documents by ID.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        # Note: FAISS doesn't support direct deletion, so we rebuild
        # This is a simplified implementation
        deleted = 0
        for doc_id in doc_ids:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                self.documents[idx] = None  # Mark as deleted
                del self.id_to_idx[doc_id]
                deleted += 1
        
        # Optionally rebuild index to reclaim space
        return deleted
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        import faiss
        
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, 'index.faiss')
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        metadata_path = os.path.join(directory, 'documents.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'id_to_idx': self.id_to_idx,
                'embedding_dim': self.embedding_dim,
                'metric': self.metric
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Saved vector store to: {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory containing saved files
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        import faiss
        
        # Load metadata
        metadata_path = os.path.join(directory, 'documents.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create instance
        store = cls(
            embedding_dim=data['embedding_dim'],
            metric=data['metric']
        )
        
        # Load FAISS index
        index_path = os.path.join(directory, 'index.faiss')
        store.index = faiss.read_index(index_path)
        
        # Restore documents
        store.documents = data['documents']
        store.id_to_idx = data['id_to_idx']
        
        print(f"Loaded vector store from: {directory}")
        print(f"Total documents: {len(store.documents)}")
        
        return store
    
    def __len__(self) -> int:
        """Return number of documents in store."""
        return len([d for d in self.documents if d is not None])
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_documents': len(self),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'index_size': self.index.ntotal if self.index else 0
        }
