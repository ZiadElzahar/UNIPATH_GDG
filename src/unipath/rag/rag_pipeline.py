"""
RAG Pipeline Module
Complete end-to-end RAG system with LLM integration.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RAGResponse:
	answer: str
	sources: List[Dict]
	context: str
	query: str
	confidence: float = 0.0

	def to_dict(self) -> Dict:
		return {
			"answer": self.answer,
			"sources": self.sources,
			"query": self.query,
			"confidence": self.confidence,
		}


class RAGPipeline:
	SYSTEM_PROMPTS = {
		"ar": """أنت مستشار أكاديمي متخصص في لوائح كلية العلوم بجامعة حلوان.
مهمتك الإجابة على أسئلة الطلاب بناءً على اللائحة الأكاديمية فقط.

قواعد مهمة:
- أجب بناءً على السياق المقدم فقط
- إذا لم تجد المعلومة في السياق، قل \"هذه المعلومة غير موجودة في اللائحة المتوفرة\"
- استخدم لغة رسمية أكاديمية
- اذكر رقم المادة إذا كان متاحاً
- كن دقيقاً وموجزاً في إجاباتك""",
		"en": """You are an academic advisor specializing in the Faculty of Science regulations at Helwan University.
Your task is to answer student questions based ONLY on the academic regulations provided.

Important rules:
- Answer based ONLY on the provided context
- If information is not in the context, say \"This information is not available in the provided regulations\"
- Use formal academic language
- Reference article numbers when available
- Be precise and concise in your answers""",
	}

	def __init__(self, retriever, llm_client=None, language: str = "ar"):
		self.retriever = retriever
		self.llm_client = llm_client
		self.language = language
		self.system_prompt = self.SYSTEM_PROMPTS.get(language, self.SYSTEM_PROMPTS["ar"])

	def query(self, question: str, k: int = 3, max_context_tokens: int = 2000, use_llm: bool = True) -> RAGResponse:
		context, sources = self.retriever.get_context(question, k=k, max_tokens=max_context_tokens)

		if not context:
			return RAGResponse(
				answer=(
					"لم أتمكن من العثور على معلومات ذات صلة بسؤالك في اللائحة."
					if self.language == "ar"
					else "I couldn't find relevant information for your question in the regulations."
				),
				sources=[],
				context="",
				query=question,
				confidence=0.0,
			)

		if use_llm and self.llm_client:
			answer = self._generate_with_llm(question, context)
		else:
			answer = self._generate_extractive(context, sources)

		return RAGResponse(
			answer=answer,
			sources=sources,
			context=context,
			query=question,
			confidence=self._calculate_confidence(sources),
		)

	def _generate_with_llm(self, question: str, context: str) -> str:
		prompt = self._build_prompt(question, context)
		try:
			return self.llm_client.generate(prompt)
		except Exception as e:
			return f"Error generating response: {str(e)}"

	def _generate_extractive(self, context: str, sources: List[Dict]) -> str:
		if not sources:
			return "لا توجد معلومات متاحة."

		answer_parts = ["بناءً على اللائحة الأكاديمية:\n\n" if self.language == "ar" else "Based on the academic regulations:\n\n"]

		for i, source in enumerate(sources[:3], 1):
			article = source.get("metadata", {}).get("article_number", "")
			text = source.get("text", "")
			if not text:
				text = context.split(f"[Document {i}]:")[-1].split("[Document")[0].strip() if context else ""
			if text:
				text = text[:600]
				if article:
					if self.language == "ar":
						answer_parts.append(f"📌 المادة ({article}):\n{text}\n\n")
					else:
						answer_parts.append(f"📌 Article ({article}):\n{text}\n\n")
				else:
					answer_parts.append(f"📌 المصدر {i}:\n{text}\n\n")

		if len(answer_parts) == 1:
			return "لم يتم العثور على نص مناسب في المصادر المسترجعة."
		return "".join(answer_parts)

	def _build_prompt(self, question: str, context: str) -> str:
		if self.language == "ar":
			return f"""{self.system_prompt}

السياق من اللائحة الأكاديمية:
{context}

سؤال الطالب: {question}

الإجابة:"""
		return f"""{self.system_prompt}

Context from Academic Regulations:
{context}

Student Question: {question}

Answer:"""

	def _calculate_confidence(self, sources: List[Dict]) -> float:
		if not sources:
			return 0.0
		scores = [s.get("score", 0) for s in sources[:3]]
		avg_score = sum(scores) / len(scores) if scores else 0
		return round(min(max(avg_score, 0), 1), 2)

	def batch_query(self, questions: List[str], k: int = 3) -> List[RAGResponse]:
		return [self.query(q, k=k) for q in questions]


class RAGSystemBuilder:
	def __init__(self, data_dir: str = "./rag_data"):
		self.data_dir = data_dir
		os.makedirs(data_dir, exist_ok=True)
		self.documents = []
		self.embedding_model = None
		self.vector_store = None
		self.retriever = None
		self.pipeline = None

	def load_from_json(self, json_path: str) -> "RAGSystemBuilder":
		with open(json_path, "r", encoding="utf-8") as f:
			data = json.load(f)

		if isinstance(data, list):
			self.documents = data
		elif isinstance(data, dict) and "chunks" in data:
			self.documents = data["chunks"]
		else:
			self.documents = [data]
		return self

	def load_from_pdf(self, pdf_path: str, chunk_size: int = 200) -> "RAGSystemBuilder":
		from .chunker import create_rag_dataset
		from .pdf_processor import PDFProcessor

		processor = PDFProcessor(pdf_path)
		full_text = processor.get_full_text()
		self.documents = create_rag_dataset(full_text, source_name=os.path.basename(pdf_path), chunk_size=chunk_size)
		return self

	def build_embeddings(self, model_name: str = "multilingual-minilm") -> "RAGSystemBuilder":
		from .embeddings import EmbeddingCache, EmbeddingModel

		self.embedding_model = EmbeddingModel(model_name)
		self.embedding_model.load_model()

		cache = EmbeddingCache(os.path.join(self.data_dir, "cache"))
		cache_id = f"docs_{len(self.documents)}"
		cached_embeddings = cache.load(cache_id)

		if cached_embeddings is not None:
			self.embeddings = cached_embeddings
		else:
			texts = [doc.get("text", "") for doc in self.documents]
			self.embeddings = self.embedding_model.encode(texts)
			cache.save(cache_id, self.embeddings)
		return self

	def build_vector_store(self) -> "RAGSystemBuilder":
		from .vector_store import FAISSVectorStore

		embedding_dim = self.embedding_model.get_embedding_dimension()
		self.vector_store = FAISSVectorStore(embedding_dim, metric="cosine")

		docs_for_store = [{"text": doc.get("text", ""), "metadata": doc.get("metadata", {})} for doc in self.documents]
		ids = [str(doc.get("chunk_id", i)) for i, doc in enumerate(self.documents)]
		self.vector_store.add(self.embeddings, docs_for_store, ids)
		self.vector_store.save(os.path.join(self.data_dir, "vector_store"))
		return self

	def build_retriever(self) -> "RAGSystemBuilder":
		from .retriever import RAGRetriever

		self.retriever = RAGRetriever(self.vector_store, self.embedding_model)
		return self

	def build_pipeline(self, llm_client=None, language: str = "ar") -> RAGPipeline:
		self.pipeline = RAGPipeline(self.retriever, llm_client=llm_client, language=language)
		return self.pipeline

	def build(self, llm_client=None, language: str = "ar") -> RAGPipeline:
		return self.build_embeddings().build_vector_store().build_retriever().build_pipeline(llm_client, language)
