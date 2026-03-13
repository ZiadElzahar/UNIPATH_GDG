"""
RAG System Tests and Examples
Demonstrates how to use the RAG system.
"""

import os
import json
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))


def test_pdf_extraction():
    """Test PDF text extraction."""
    print("\n" + "=" * 60)
    print("Testing PDF Extraction")
    print("=" * 60)
    
    from src.unipath.rag.pdf_processor import PDFProcessor
    
    pdf_path = Path(__file__).parent / "data" / "لائحة الساعات المعتمدة كلية العلوم جامعة حلوان 2021 النسخة النهائية للقرار الوزارى.pdf"
    
    if not pdf_path.exists():
        print(f"❌ PDF not found at: {pdf_path}")
        return False
    
    processor = PDFProcessor(str(pdf_path))
    pages = processor.extract_text()
    
    print(f"✓ Extracted {len(pages)} pages")
    print(f"  First page preview: {pages[0].text[:200]}...")
    
    # Test section extraction
    sections = processor.extract_sections()
    print(f"✓ Found {len(sections)} articles/sections")
    
    return True


def test_chunking():
    """Test text chunking."""
    print("\n" + "=" * 60)
    print("Testing Text Chunking")
    print("=" * 60)
    
    from src.unipath.rag.chunker import SemanticChunker, create_rag_dataset
    
    sample_text = """
    مادة (1): نظام الدراسة
    تمنح كلية العلوم جامعة حلوان درجة البكالوريوس في العلوم بعد دراسة واجتياز الطالب بنجاح لعدد 138 ساعة معتمدة.
    
    مادة (2): لغة التدريس وأسلوبه
    لغة التدريس والامتحان هي اللغة الإنجليزية. تنتهج الكلية أساليب تدريس حديثة ومتنوعة.
    
    مادة (3): معيار الساعة المعتمدة
    بالنسبة للمحاضرات النظرية: تحتسب ساعة معتمدة واحدة لكل محاضرة مدتها ساعة نظرية واحدة أسبوعياً.
    """
    
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_text, {'source': 'test'})
    
    print(f"✓ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i+1}: {len(chunk.text.split())} words")
        print(f"    Preview: {chunk.text[:100]}...")
    
    return True


def test_embeddings():
    """Test embedding generation."""
    print("\n" + "=" * 60)
    print("Testing Embeddings")
    print("=" * 60)
    
    from src.unipath.rag.embeddings import EmbeddingModel
    
    model = EmbeddingModel('multilingual-minilm')
    model.load_model()
    
    texts = [
        "ما هي شروط التخرج؟",
        "كم عدد الساعات المعتمدة؟",
        "What are the graduation requirements?"
    ]
    
    embeddings = model.encode(texts, show_progress=False)
    
    print(f"✓ Generated embeddings with shape: {embeddings.shape}")
    
    # Test similarity
    sim = model.similarity(embeddings[0], embeddings[1])
    print(f"  Similarity (Q1 vs Q2): {sim:.3f}")
    
    sim = model.similarity(embeddings[0], embeddings[2])
    print(f"  Similarity (Arabic vs English): {sim:.3f}")
    
    return True


def test_vector_store():
    """Test vector store operations."""
    print("\n" + "=" * 60)
    print("Testing Vector Store")
    print("=" * 60)
    
    from src.unipath.rag.embeddings import EmbeddingModel
    from src.unipath.rag.vector_store import FAISSVectorStore
    
    # Create embeddings
    model = EmbeddingModel('multilingual-minilm')
    model.load_model()
    
    documents = [
        {"text": "شروط التخرج تتطلب إتمام 138 ساعة معتمدة", "metadata": {"article": 1}},
        {"text": "يتكون العام الدراسي من فصلين دراسيين", "metadata": {"article": 5}},
        {"text": "لغة التدريس هي اللغة الإنجليزية", "metadata": {"article": 2}},
    ]
    
    texts = [d['text'] for d in documents]
    embeddings = model.encode(texts, show_progress=False)
    
    # Create and populate vector store
    store = FAISSVectorStore(embedding_dim=model.get_embedding_dimension())
    store.add(embeddings, documents)
    
    print(f"✓ Added {len(store)} documents to vector store")
    
    # Test search
    query = "كم عدد الساعات المطلوبة للتخرج؟"
    query_emb = model.encode(query, show_progress=False)
    results = store.search(query_emb, k=2)
    
    print(f"✓ Search results for '{query}':")
    for i, r in enumerate(results):
        print(f"  {i+1}. Score: {r.score:.3f} - {r.text[:50]}...")
    
    return True


def test_retrieval():
    """Test the full retrieval pipeline."""
    print("\n" + "=" * 60)
    print("Testing Retrieval Pipeline")
    print("=" * 60)
    
    from src.unipath.rag.embeddings import EmbeddingModel
    from src.unipath.rag.vector_store import FAISSVectorStore
    from src.unipath.rag.retriever import RAGRetriever
    
    # Load existing data
    data_path = Path(__file__).parent / "data" / "rag_dataset_fixed.json"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return False
    
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)[:50]  # Use subset for testing
    
    # Build components
    model = EmbeddingModel('multilingual-minilm')
    model.load_model()
    
    texts = [d.get('text', '') for d in documents]
    embeddings = model.encode(texts, show_progress=True)
    
    store = FAISSVectorStore(embedding_dim=model.get_embedding_dimension())
    store.add(embeddings, [{'text': d.get('text'), 'metadata': d.get('metadata', {})} for d in documents])
    
    retriever = RAGRetriever(store, model)
    
    # Test queries
    queries = [
        "ما هي شروط التخرج؟",
        "ما هي مستويات الدراسة؟",
        "ما هو نظام الفصل الصيفي؟"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.retrieve(query, k=2)
        
        for i, r in enumerate(results):
            print(f"  {i+1}. Score: {r.score:.3f}")
            print(f"     Text: {r.text[:100]}...")
    
    return True


def test_full_pipeline():
    """Test the complete RAG pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full RAG Pipeline")
    print("=" * 60)
    
    from src.unipath.rag.rag_pipeline import RAGSystemBuilder
    
    data_path = Path(__file__).parent / "data" / "rag_dataset_fixed.json"
    rag_data_dir = Path(__file__).parent / "rag_data_test"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return False
    
    # Build system
    builder = RAGSystemBuilder(data_dir=str(rag_data_dir))
    builder.load_from_json(str(data_path))
    
    # Limit documents for testing
    builder.documents = builder.documents[:100]
    
    pipeline = builder.build(llm_client=None, language='ar')
    
    # Test queries
    queries = [
        "ما هي شروط التخرج؟",
        "كم عدد الساعات المعتمدة المطلوبة؟",
        "ما هي البرامج المتاحة في قسم الرياضيات؟"
    ]
    
    for query in queries:
        print(f"\n📝 Question: {query}")
        response = pipeline.query(query, k=3)
        print(f"📖 Answer: {response.answer[:200]}...")
        print(f"💯 Confidence: {response.confidence:.0%}")
        print(f"📚 Sources: {len(response.sources)}")
    
    # Cleanup test directory
    import shutil
    if rag_data_dir.exists():
        shutil.rmtree(rag_data_dir)
    
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RAG SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("PDF Extraction", test_pdf_extraction),
        ("Chunking", test_chunking),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("Retrieval", test_retrieval),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        test_map = {
            "pdf": test_pdf_extraction,
            "chunk": test_chunking,
            "embed": test_embeddings,
            "store": test_vector_store,
            "retrieve": test_retrieval,
            "pipeline": test_full_pipeline,
        }
        
        if test_name in test_map:
            test_map[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available: {list(test_map.keys())}")
    else:
        run_all_tests()
