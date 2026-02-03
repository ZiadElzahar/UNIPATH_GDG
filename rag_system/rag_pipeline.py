"""
RAG Pipeline Module
Complete end-to-end RAG system with LLM integration.
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: List[Dict]
    context: str
    query: str
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'answer': self.answer,
            'sources': self.sources,
            'query': self.query,
            'confidence': self.confidence
        }


class RAGPipeline:
    """
    Complete RAG pipeline for academic regulation queries.
    Integrates retrieval with LLM generation.
    """
    
    # System prompts for different languages
    SYSTEM_PROMPTS = {
        'ar': """أنت مستشار أكاديمي متخصص في لوائح كلية العلوم بجامعة حلوان.
مهمتك الإجابة على أسئلة الطلاب بناءً على اللائحة الأكاديمية فقط.

قواعد مهمة:
- أجب بناءً على السياق المقدم فقط
- إذا لم تجد المعلومة في السياق، قل "هذه المعلومة غير موجودة في اللائحة المتوفرة"
- استخدم لغة رسمية أكاديمية
- اذكر رقم المادة إذا كان متاحاً
- كن دقيقاً وموجزاً في إجاباتك""",
        
        'en': """You are an academic advisor specializing in the Faculty of Science regulations at Helwan University.
Your task is to answer student questions based ONLY on the academic regulations provided.

Important rules:
- Answer based ONLY on the provided context
- If information is not in the context, say "This information is not available in the provided regulations"
- Use formal academic language
- Reference article numbers when available
- Be precise and concise in your answers"""
    }
    
    def __init__(
        self,
        retriever,
        llm_client=None,
        language: str = 'ar'
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: RAGRetriever instance
            llm_client: LLM client with generate() method (optional)
            language: Primary language ('ar' or 'en')
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.language = language
        self.system_prompt = self.SYSTEM_PROMPTS.get(language, self.SYSTEM_PROMPTS['ar'])
    
    def query(
        self,
        question: str,
        k: int = 3,
        max_context_tokens: int = 2000,
        use_llm: bool = True
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            max_context_tokens: Maximum context length
            use_llm: Whether to use LLM for generation
            
        Returns:
            RAGResponse object
        """
        # Retrieve relevant context
        context, sources = self.retriever.get_context(
            question,
            k=k,
            max_tokens=max_context_tokens
        )
        
        if not context:
            return RAGResponse(
                answer="لم أتمكن من العثور على معلومات ذات صلة بسؤالك في اللائحة."
                       if self.language == 'ar' else
                       "I couldn't find relevant information for your question in the regulations.",
                sources=[],
                context="",
                query=question,
                confidence=0.0
            )
        
        # Generate answer
        if use_llm and self.llm_client:
            answer = self._generate_with_llm(question, context)
        else:
            answer = self._generate_extractive(question, context, sources)
        
        # Calculate confidence based on retrieval scores
        confidence = self._calculate_confidence(sources)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            context=context,
            query=question,
            confidence=confidence
        )
    
    def _generate_with_llm(self, question: str, context: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        prompt = self._build_prompt(question, context)
        
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_extractive(
        self,
        question: str,
        context: str,
        sources: List[Dict]
    ) -> str:
        """
        Generate a simple extractive answer without LLM.
        
        Args:
            question: User question
            context: Retrieved context
            sources: Source documents
            
        Returns:
            Extractive answer
        """
        # For now, return the most relevant chunk with formatting
        if not sources:
            return "لا توجد معلومات متاحة."
        
        # Build a simple answer from retrieved chunks
        answer_parts = []
        
        if self.language == 'ar':
            answer_parts.append("بناءً على اللائحة الأكاديمية:\n\n")
        else:
            answer_parts.append("Based on the academic regulations:\n\n")
        
        for i, source in enumerate(sources[:3], 1):
            article = source.get('metadata', {}).get('article_number', '')
            
            # Get text from source or context
            text = source.get('text', '')
            if not text:
                # Fallback: try to extract from context
                text = context.split(f"[Document {i}]:")[-1].split("[Document")[0].strip() if context else ''
            
            if text:
                text = text[:600]  # Limit length
                if article:
                    if self.language == 'ar':
                        answer_parts.append(f"📌 المادة ({article}):\n{text}\n\n")
                    else:
                        answer_parts.append(f"📌 Article ({article}):\n{text}\n\n")
                else:
                    answer_parts.append(f"📌 المصدر {i}:\n{text}\n\n")
        
        if len(answer_parts) == 1:  # Only header, no content
            return "لم يتم العثور على نص مناسب في المصادر المسترجعة."
        
        return "".join(answer_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the prompt for the LLM.
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        if self.language == 'ar':
            prompt = f"""{self.system_prompt}

السياق من اللائحة الأكاديمية:
{context}

سؤال الطالب: {question}

الإجابة:"""
        else:
            prompt = f"""{self.system_prompt}

Context from Academic Regulations:
{context}

Student Question: {question}

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, sources: List[Dict]) -> float:
        """
        Calculate confidence score based on retrieval quality.
        
        Args:
            sources: Retrieved sources with scores
            
        Returns:
            Confidence score between 0 and 1
        """
        if not sources:
            return 0.0
        
        # Average of top source scores
        scores = [s.get('score', 0) for s in sources[:3]]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Normalize to 0-1 range
        confidence = min(max(avg_score, 0), 1)
        
        return round(confidence, 2)
    
    def batch_query(
        self,
        questions: List[str],
        k: int = 3
    ) -> List[RAGResponse]:
        """
        Process multiple queries.
        
        Args:
            questions: List of questions
            k: Number of documents per query
            
        Returns:
            List of RAGResponse objects
        """
        return [self.query(q, k=k) for q in questions]


class RAGSystemBuilder:
    """
    Builder class for constructing a complete RAG system from scratch.
    """
    
    def __init__(self, data_dir: str = './rag_data'):
        """
        Initialize the builder.
        
        Args:
            data_dir: Directory for storing RAG system data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.documents = []
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.pipeline = None
    
    def load_from_json(self, json_path: str) -> 'RAGSystemBuilder':
        """
        Load documents from a JSON file.
        
        Args:
            json_path: Path to JSON file with documents
            
        Returns:
            self for chaining
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.documents = data
        elif isinstance(data, dict) and 'chunks' in data:
            self.documents = data['chunks']
        else:
            self.documents = [data]
        
        print(f"Loaded {len(self.documents)} documents from {json_path}")
        return self
    
    def load_from_pdf(self, pdf_path: str, chunk_size: int = 200) -> 'RAGSystemBuilder':
        """
        Load and process a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Target words per chunk
            
        Returns:
            self for chaining
        """
        from .pdf_processor import PDFProcessor
        from .chunker import create_rag_dataset
        
        processor = PDFProcessor(pdf_path)
        full_text = processor.get_full_text()
        
        self.documents = create_rag_dataset(
            full_text,
            source_name=os.path.basename(pdf_path),
            chunk_size=chunk_size
        )
        
        print(f"Created {len(self.documents)} chunks from PDF")
        return self
    
    def build_embeddings(
        self,
        model_name: str = 'multilingual-minilm'
    ) -> 'RAGSystemBuilder':
        """
        Build embeddings for all documents.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            self for chaining
        """
        from .embeddings import EmbeddingModel, EmbeddingCache
        
        self.embedding_model = EmbeddingModel(model_name)
        self.embedding_model.load_model()
        
        # Check cache
        cache = EmbeddingCache(os.path.join(self.data_dir, 'cache'))
        cache_id = f"docs_{len(self.documents)}"
        
        cached_embeddings = cache.load(cache_id)
        
        if cached_embeddings is not None:
            self.embeddings = cached_embeddings
        else:
            texts = [doc.get('text', '') for doc in self.documents]
            self.embeddings = self.embedding_model.encode(texts)
            cache.save(cache_id, self.embeddings)
        
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
        return self
    
    def build_vector_store(self) -> 'RAGSystemBuilder':
        """
        Build the vector store with embeddings.
        
        Returns:
            self for chaining
        """
        from .vector_store import FAISSVectorStore
        
        embedding_dim = self.embedding_model.get_embedding_dimension()
        self.vector_store = FAISSVectorStore(embedding_dim, metric='cosine')
        
        # Prepare documents for storage
        docs_for_store = [
            {
                'text': doc.get('text', ''),
                'metadata': doc.get('metadata', {})
            }
            for doc in self.documents
        ]
        
        ids = [str(doc.get('chunk_id', i)) for i, doc in enumerate(self.documents)]
        
        self.vector_store.add(self.embeddings, docs_for_store, ids)
        
        # Save vector store
        self.vector_store.save(os.path.join(self.data_dir, 'vector_store'))
        
        print(f"Built vector store with {len(self.vector_store)} documents")
        return self
    
    def build_retriever(self) -> 'RAGSystemBuilder':
        """
        Build the retriever.
        
        Returns:
            self for chaining
        """
        from .retriever import RAGRetriever
        
        self.retriever = RAGRetriever(
            self.vector_store,
            self.embedding_model
        )
        
        print("Built retriever")
        return self
    
    def build_pipeline(
        self,
        llm_client=None,
        language: str = 'ar'
    ) -> RAGPipeline:
        """
        Build the complete RAG pipeline.
        
        Args:
            llm_client: Optional LLM client
            language: Primary language
            
        Returns:
            RAGPipeline instance
        """
        self.pipeline = RAGPipeline(
            self.retriever,
            llm_client=llm_client,
            language=language
        )
        
        print("Built complete RAG pipeline")
        return self.pipeline
    
    def build(self, llm_client=None, language: str = 'ar') -> RAGPipeline:
        """
        Build all components and return the pipeline.
        
        Args:
            llm_client: Optional LLM client
            language: Primary language
            
        Returns:
            RAGPipeline instance
        """
        return (
            self
            .build_embeddings()
            .build_vector_store()
            .build_retriever()
            .build_pipeline(llm_client, language)
        )
