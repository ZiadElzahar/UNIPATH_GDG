"""
Main RAG Application
Quick start script for building and using the RAG system.
"""

import os
import json
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RAG_DATA_DIR = PROJECT_DIR / "rag_data"
PDF_PATH = PROJECT_DIR / "data" / "لائحة الساعات المعتمدة كلية العلوم جامعة حلوان 2021 النسخة النهائية للقرار الوزارى.pdf"


def build_rag_system_from_existing_json():
    """
    Build RAG system from existing rag_dataset_fixed.json
    """
    from src.unipath.rag import RAGSystemBuilder
    
    json_path = DATA_DIR / "rag_dataset_fixed.json"
    
    if not json_path.exists():
        print(f"JSON file not found at {json_path}")
        return None
    
    print("=" * 60)
    print("Building RAG System from existing JSON dataset")
    print("=" * 60)
    
    builder = RAGSystemBuilder(data_dir=str(RAG_DATA_DIR))
    
    # Load documents
    builder.load_from_json(str(json_path))
    
    # Build all components
    pipeline = builder.build(llm_client=None, language='ar')
    
    print("\n✓ RAG System built successfully!")
    print(f"  - Documents indexed: {len(builder.documents)}")
    print(f"  - Vector store saved to: {RAG_DATA_DIR / 'vector_store'}")
    
    return pipeline


def build_rag_system_from_pdf():
    """
    Build RAG system directly from PDF file.
    """
    from src.unipath.rag import RAGSystemBuilder
    
    if not PDF_PATH.exists():
        print(f"PDF file not found at {PDF_PATH}")
        return None
    
    print("=" * 60)
    print("Building RAG System from PDF")
    print("=" * 60)
    
    builder = RAGSystemBuilder(data_dir=str(RAG_DATA_DIR))
    
    # Load and process PDF
    builder.load_from_pdf(str(PDF_PATH), chunk_size=200)
    
    # Build all components
    pipeline = builder.build(llm_client=None, language='ar')
    
    print("\n✓ RAG System built successfully!")
    
    return pipeline


def load_existing_rag_system():
    """
    Load a previously built RAG system.
    """
    from src.unipath.rag.embeddings import EmbeddingModel
    from src.unipath.rag.vector_store import FAISSVectorStore
    from src.unipath.rag.retriever import RAGRetriever
    from src.unipath.rag.rag_pipeline import RAGPipeline
    
    vector_store_path = RAG_DATA_DIR / "vector_store"
    
    if not vector_store_path.exists():
        print("No existing RAG system found. Building new one...")
        return build_rag_system_from_existing_json()
    
    print("Loading existing RAG system...")
    
    # Load components
    embedding_model = EmbeddingModel('multilingual-minilm')
    embedding_model.load_model()
    
    vector_store = FAISSVectorStore.load(str(vector_store_path))
    
    retriever = RAGRetriever(vector_store, embedding_model)
    
    pipeline = RAGPipeline(retriever, language='ar')
    
    print("✓ RAG System loaded successfully!")
    
    return pipeline


def interactive_query(pipeline):
    """
    Interactive query loop.
    """
    print("\n" + "=" * 60)
    print("RAG System Ready - Academic Regulations Q&A")
    print("Type 'quit' to exit, 'help' for example questions")
    print("=" * 60 + "\n")
    
    example_questions = [
        "ما هي شروط التخرج؟",
        "كم عدد الساعات المعتمدة للتخرج؟",
        "ما هي مستويات الدراسة؟",
        "ما هو نظام الفصل الصيفي؟",
        "ما هي متطلبات التسجيل؟",
        "كيف يتم حساب المعدل التراكمي؟",
        "ما هي البرامج المتاحة في قسم الرياضيات؟",
        "ما هي شروط الحصول على مرتبة الشرف؟"
    ]
    
    while True:
        try:
            query = input("\n📝 سؤالك: ").strip()
            
            if query.lower() == 'quit':
                print("مع السلامة!")
                break
            
            if query.lower() == 'help':
                print("\nأمثلة على الأسئلة:")
                for i, q in enumerate(example_questions, 1):
                    print(f"  {i}. {q}")
                continue
            
            if not query:
                continue
            
            # Process query
            print("\n🔍 جاري البحث...")
            response = pipeline.query(query, k=3)
            
            # Display results
            print("\n" + "-" * 50)
            print("📖 الإجابة:")
            print("-" * 50)
            print(response.answer)
            
            if response.sources:
                print("\n📚 المصادر:")
                for i, source in enumerate(response.sources, 1):
                    article = source.get('metadata', {}).get('article_number', 'N/A')
                    score = source.get('score', 0)
                    print(f"  {i}. مادة {article} (relevance: {score:.2f})")
            
            print(f"\n💯 Confidence: {response.confidence:.0%}")
            
        except KeyboardInterrupt:
            print("\n\nمع السلامة!")
            break


def test_retrieval(pipeline, queries=None):
    """
    Test retrieval with sample queries.
    """
    if queries is None:
        queries = [
            "ما هي شروط التخرج؟",
            "كم عدد الساعات المعتمدة المطلوبة؟",
            "ما هو نظام الدراسة؟",
            "ما هي الأقسام العلمية؟"
        ]
    
    print("\n" + "=" * 60)
    print("Testing RAG Retrieval")
    print("=" * 60)
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        response = pipeline.query(query, k=2)
        
        print(f"Answer preview: {response.answer[:200]}...")
        print(f"Sources: {len(response.sources)}")
        print(f"Confidence: {response.confidence:.0%}")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "build":
            # Build from scratch
            pipeline = build_rag_system_from_existing_json()
        elif command == "build-pdf":
            # Build from PDF
            pipeline = build_rag_system_from_pdf()
        elif command == "test":
            # Test mode
            pipeline = load_existing_rag_system()
            if pipeline:
                test_retrieval(pipeline)
            sys.exit(0)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python rag_app.py [build|build-pdf|test]")
            sys.exit(1)
    else:
        # Default: load existing or build
        pipeline = load_existing_rag_system()
    
    if pipeline:
        interactive_query(pipeline)
