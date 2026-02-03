"""
Embedding Model Module
Handles text embedding generation using sentence transformers.
Optimized for Arabic/English multilingual support.
"""

import numpy as np
from typing import List, Union, Optional
import os


class EmbeddingModel:
    """
    Generates text embeddings using sentence transformers.
    Supports multilingual models for Arabic content.
    """
    
    # Recommended models for Arabic text
    ARABIC_MODELS = {
        'multilingual-minilm': 'paraphrase-multilingual-MiniLM-L12-v2',
        'multilingual-mpnet': 'paraphrase-multilingual-mpnet-base-v2',
        'arabic-bert': 'aubmindlab/bert-base-arabertv2',
        'labse': 'sentence-transformers/LaBSE',  # Best for Arabic
    }
    
    def __init__(
        self,
        model_name: str = 'multilingual-minilm',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Model identifier (key from ARABIC_MODELS or full HuggingFace name)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            cache_dir: Directory to cache model files
        """
        self.model_name = self.ARABIC_MODELS.get(model_name, model_name)
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.embedding_dim = None
        
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        kwargs = {}
        if self.cache_dir:
            kwargs['cache_folder'] = self.cache_dir
        if self.device:
            kwargs['device'] = self.device
            
        self.model = SentenceTransformer(self.model_name, **kwargs)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"Loaded model: {self.model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
            
        Returns:
            NumPy array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        texts = [self._preprocess(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )
        
        return np.array(embeddings).astype('float32')
    
    def _preprocess(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (most models have 512 token limit)
        if len(text) > 5000:
            text = text[:5000]
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self.embedding_dim is None:
            self.load_model()
        return self.embedding_dim
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize if not already
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        return 0.0
    
    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarity between query and multiple documents.
        
        Args:
            query_embedding: Query embedding vector (1D)
            document_embeddings: Document embedding matrix (2D)
            
        Returns:
            Array of similarity scores
        """
        # Ensure proper shapes
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.dot(query_norm, doc_norms.T).flatten()
        
        return similarities


class EmbeddingCache:
    """
    Caches embeddings to disk for faster subsequent loads.
    """
    
    def __init__(self, cache_dir: str = './embedding_cache'):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, identifier: str) -> str:
        """Get the cache file path for an identifier."""
        return os.path.join(self.cache_dir, f"{identifier}_embeddings.npy")
    
    def save(self, identifier: str, embeddings: np.ndarray) -> None:
        """
        Save embeddings to cache.
        
        Args:
            identifier: Unique identifier for the embeddings
            embeddings: NumPy array of embeddings
        """
        cache_path = self.get_cache_path(identifier)
        np.save(cache_path, embeddings)
        print(f"Cached embeddings to: {cache_path}")
    
    def load(self, identifier: str) -> Optional[np.ndarray]:
        """
        Load embeddings from cache.
        
        Args:
            identifier: Unique identifier for the embeddings
            
        Returns:
            NumPy array of embeddings, or None if not cached
        """
        cache_path = self.get_cache_path(identifier)
        
        if os.path.exists(cache_path):
            embeddings = np.load(cache_path)
            print(f"Loaded embeddings from cache: {cache_path}")
            return embeddings
        
        return None
    
    def exists(self, identifier: str) -> bool:
        """Check if embeddings are cached."""
        return os.path.exists(self.get_cache_path(identifier))
