"""
Retriever Module
Handles document retrieval with reranking and query enhancement.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class RetrievalResult:
    """Result from retrieval with all relevant information."""
    text: str
    score: float
    metadata: Dict
    chunk_id: str
    rerank_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'score': self.score,
            'rerank_score': self.rerank_score,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id
        }


class RAGRetriever:
    """
    Advanced retriever with query enhancement, reranking, and hybrid search.
    """
    
    def __init__(
        self,
        vector_store,
        embedding_model,
        reranker=None
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: FAISSVectorStore instance
            embedding_model: EmbeddingModel instance
            reranker: Optional reranker model
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = reranker
        
        # Arabic query enhancement patterns
        self.arabic_stop_words = {
            'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه',
            'التي', 'الذي', 'ما', 'هل', 'كيف', 'أين', 'متى', 'لماذا',
            'و', 'أو', 'ثم', 'لكن', 'بل', 'حتى', 'إذا', 'إن', 'أن',
            'كان', 'يكون', 'تكون', 'كانت', 'كانوا'
        }
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3,
        rerank: bool = True,
        expand_query: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold
            rerank: Whether to apply reranking
            expand_query: Whether to expand the query
            
        Returns:
            List of RetrievalResult objects
        """
        # Query preprocessing
        processed_query = self._preprocess_query(query)
        
        # Optional query expansion
        if expand_query:
            expanded_queries = self._expand_query(processed_query)
        else:
            expanded_queries = [processed_query]
        
        # Retrieve candidates
        all_results = []
        seen_chunks = set()
        
        for q in expanded_queries:
            query_embedding = self.embedding_model.encode(q, show_progress=False)
            results = self.vector_store.search(query_embedding, k=k * 2, threshold=threshold)
            
            for result in results:
                if result.chunk_id not in seen_chunks:
                    all_results.append(RetrievalResult(
                        text=result.text,
                        score=result.score,
                        metadata=result.metadata,
                        chunk_id=result.chunk_id
                    ))
                    seen_chunks.add(result.chunk_id)
        
        # Sort by initial score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply reranking if enabled
        if rerank and self.reranker and len(all_results) > 0:
            all_results = self._rerank(query, all_results[:k * 2])
        
        return all_results[:k]
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query for better matching.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query
        """
        # Normalize Arabic characters
        query = query.strip()
        
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        # Normalize some Arabic variations
        query = query.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        
        return query
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and variations.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        queries = [query]
        
        # Academic term expansions (Arabic)
        expansions = {
            'شروط': ['متطلبات', 'requirements'],
            'تخرج': ['graduation', 'إنهاء الدراسة'],
            'مقرر': ['course', 'مادة دراسية'],
            'ساعات': ['credits', 'وحدات'],
            'معتمدة': ['credit hours', 'معتمده'],
            'تسجيل': ['registration', 'قيد'],
            'حذف': ['drop', 'إسقاط'],
            'إضافة': ['add', 'إلحاق'],
            'معدل': ['GPA', 'تقدير تراكمي'],
            'فصل': ['semester', 'ترم'],
            'رسوب': ['fail', 'إخفاق'],
            'نجاح': ['pass', 'اجتياز'],
        }
        
        # Add variations if key terms are found
        for term, alternatives in expansions.items():
            if term in query.lower():
                for alt in alternatives:
                    new_query = query.replace(term, alt)
                    if new_query != query:
                        queries.append(new_query)
        
        return queries[:3]  # Limit expansions
    
    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank results using a cross-encoder or other reranking method.
        
        Args:
            query: Original query
            results: Initial retrieval results
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        # Simple keyword-based reranking if no reranker model
        if self.reranker is None:
            return self._keyword_rerank(query, results)
        
        # Use cross-encoder reranker
        pairs = [(query, r.text) for r in results]
        rerank_scores = self.reranker.predict(pairs)
        
        for result, score in zip(results, rerank_scores):
            result.rerank_score = float(score)
        
        results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return results
    
    def _keyword_rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Simple keyword-based reranking.
        
        Args:
            query: Original query
            results: Initial results
            
        Returns:
            Reranked results
        """
        # Extract keywords from query
        query_words = set(query.lower().split()) - self.arabic_stop_words
        
        for result in results:
            text_lower = result.text.lower()
            
            # Count keyword matches
            matches = sum(1 for word in query_words if word in text_lower)
            keyword_score = matches / max(len(query_words), 1)
            
            # Combine with semantic score
            combined_score = 0.7 * result.score + 0.3 * keyword_score
            result.rerank_score = combined_score
        
        results.sort(key=lambda x: x.rerank_score or x.score, reverse=True)
        return results
    
    def retrieve_by_article(
        self,
        article_number: int,
        k: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks from a specific article.
        
        Args:
            article_number: Article number to retrieve
            k: Maximum number of chunks
            
        Returns:
            List of results from that article
        """
        results = []
        
        for doc in self.vector_store.documents:
            if doc and doc.get('metadata', {}).get('article_number') == article_number:
                results.append(RetrievalResult(
                    text=doc['text'],
                    score=1.0,
                    metadata=doc['metadata'],
                    chunk_id=doc['chunk_id']
                ))
        
        return results[:k]
    
    def get_context(
        self,
        query: str,
        k: int = 3,
        max_tokens: int = 2000
    ) -> Tuple[str, List[Dict]]:
        """
        Get formatted context for LLM prompt.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            max_tokens: Maximum approximate tokens for context
            
        Returns:
            Tuple of (formatted context string, list of sources)
        """
        results = self.retrieve(query, k=k)
        
        context_parts = []
        sources = []
        total_length = 0
        
        for i, result in enumerate(results):
            # Estimate tokens (rough approximation)
            text_length = len(result.text.split())
            
            if total_length + text_length > max_tokens:
                break
            
            context_parts.append(f"[Document {i+1}]:\n{result.text}")
            sources.append({
                'chunk_id': result.chunk_id,
                'score': result.score,
                'metadata': result.metadata,
                'text': result.text  # Include the actual text content
            })
            total_length += text_length
        
        context = "\n\n".join(context_parts)
        
        return context, sources
