"""
RAG System Benchmark & Status Report Generator
Measures accuracy, latency, retrieval quality, and system stats.
"""

import os
import sys
import json
import time
import statistics
import traceback
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RAG_DATA_DIR = PROJECT_DIR / "rag_data"

# ─── Test Questions with Expected Keywords (Ground Truth) ──────────────
# Each test: (question, expected_keywords_in_answer, category)
TEST_CASES = [
    # Graduation requirements
    (
        "ما هي شروط التخرج؟",
        ["138", "ساعة", "معتمدة"],
        "Graduation Requirements"
    ),
    (
        "كم عدد الساعات المعتمدة المطلوبة للتخرج؟",
        ["138", "ساعة"],
        "Graduation Requirements"
    ),
    # Study levels
    (
        "ما هي مستويات الدراسة؟",
        ["أربعة", "مستوي", "Freshman", "Sophomore", "Junior", "Senior"],
        "Study Levels"
    ),
    (
        "ما هو المستوى الأول؟",
        ["صفر", "30", "Freshman"],
        "Study Levels"
    ),
    # Study system
    (
        "ما هو نظام الدراسة؟",
        ["بكالوريوس", "ساعة", "معتمدة"],
        "Study System"
    ),
    # Teaching language
    (
        "ما هي لغة التدريس؟",
        ["الإنجليزية", "نجليزية", "الانجليزية", "English"],
        "Teaching Language"
    ),
    # Academic year
    (
        "ما هو العام الدراسي؟",
        ["فصل", "دراسي", "الخريف", "الربيع"],
        "Academic Year"
    ),
    # Summer semester
    (
        "ما هو نظام الفصل الصيفي؟",
        ["صيفي", "ثمانية", "8", "أسابيع"],
        "Summer Semester"
    ),
    # Departments
    (
        "ما هي الأقسام العلمية؟",
        ["الرياضيات", "الفيزياء", "الكيمياء", "الجيولوجيا"],
        "Departments"
    ),
    # GPA
    (
        "كيف يتم حساب المعدل التراكمي؟",
        ["معدل", "تراكم", "GPA"],
        "GPA Calculation"
    ),
    # Registration
    (
        "ما هي متطلبات التسجيل؟",
        ["تسجيل", "مستوى", "ساعة"],
        "Registration"
    ),
    # Honor
    (
        "ما هي شروط الحصول على مرتبة الشرف؟",
        ["شرف", "تقدير", "ممتاز"],
        "Honors"
    ),
    # Study load
    (
        "ما هو العبء الدراسي؟",
        ["ساعة", "عبء"],
        "Study Load"
    ),
    # Credit hour definition
    (
        "ما هو معيار الساعة المعتمدة؟",
        ["ساعة", "معتمدة", "محاضرة", "نظرية"],
        "Credit Hour"
    ),
    # Programs
    (
        "ما هي البرامج المتاحة في قسم الرياضيات؟",
        ["رياضيات", "حاسب", "إحصاء", "برنامج"],
        "Programs"
    ),
]


def load_rag_system():
    """Load the existing RAG system."""
    from src.unipath.rag.embeddings import EmbeddingModel
    from src.unipath.rag.vector_store import FAISSVectorStore
    from src.unipath.rag.retriever import RAGRetriever
    from src.unipath.rag.rag_pipeline import RAGPipeline

    vector_store_path = RAG_DATA_DIR / "vector_store"

    if not vector_store_path.exists():
        print("[!] No existing vector store found. Building from JSON...")
        from src.unipath.rag.rag_pipeline import RAGSystemBuilder
        builder = RAGSystemBuilder(data_dir=str(RAG_DATA_DIR))
        json_path = DATA_DIR / "rag_dataset_fixed.json"
        builder.load_from_json(str(json_path))
        pipeline = builder.build(llm_client=None, language='ar')
        return pipeline, builder.documents, builder.embedding_model

    embedding_model = EmbeddingModel('multilingual-minilm')
    embedding_model.load_model()

    vector_store = FAISSVectorStore.load(str(vector_store_path))

    retriever = RAGRetriever(vector_store, embedding_model)
    pipeline = RAGPipeline(retriever, language='ar')

    # Load documents for stats
    json_path = DATA_DIR / "rag_dataset_fixed.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    return pipeline, documents, embedding_model


def measure_latency(pipeline, query, k=3, runs=3):
    """Measure query latency over multiple runs. Returns (avg_ms, min_ms, max_ms, response)."""
    times = []
    response = None
    for _ in range(runs):
        start = time.perf_counter()
        response = pipeline.query(query, k=k)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    return statistics.mean(times), min(times), max(times), response


def check_keyword_hit(answer_text, expected_keywords):
    """Check how many expected keywords appear in the answer. Returns (hits, total, matched, missed)."""
    answer_lower = answer_text.lower()
    matched = []
    missed = []
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            matched.append(kw)
        else:
            missed.append(kw)
    return len(matched), len(expected_keywords), matched, missed


def measure_embedding_latency(embedding_model, texts, runs=3):
    """Measure embedding generation latency."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        embedding_model.encode(texts, show_progress=False)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return statistics.mean(times), min(times), max(times)


def run_benchmark():
    """Run full benchmark and return results dict."""
    print("=" * 60)
    print("  RAG System Benchmark")
    print("=" * 60)

    # ── Load system ──────────────────────────────────────────
    print("\n[1/5] Loading RAG system...")
    load_start = time.perf_counter()
    pipeline, documents, embedding_model = load_rag_system()
    load_time = (time.perf_counter() - load_start) * 1000
    print(f"      Loaded in {load_time:.0f} ms")

    num_docs = len(documents)
    vector_store = pipeline.retriever.vector_store
    num_indexed = len(vector_store) if hasattr(vector_store, '__len__') else "N/A"
    embedding_dim = embedding_model.embedding_dim

    # ── Dataset stats ────────────────────────────────────────
    print("\n[2/5] Computing dataset stats...")
    word_counts = []
    for doc in documents:
        text = doc.get('text', '')
        word_counts.append(len(text.split()))
    avg_words = statistics.mean(word_counts) if word_counts else 0
    median_words = statistics.median(word_counts) if word_counts else 0
    total_words = sum(word_counts)

    # ── Embedding latency ────────────────────────────────────
    print("\n[3/5] Measuring embedding latency...")
    sample_texts = [
        "ما هي شروط التخرج؟",
        "كم عدد الساعات المعتمدة المطلوبة؟",
        "ما هو نظام الفصل الصيفي؟",
        "ما هي البرامج المتاحة في قسم الفيزياء؟",
        "كيف يتم حساب المعدل التراكمي؟",
    ]
    emb_avg, emb_min, emb_max = measure_embedding_latency(embedding_model, sample_texts)
    single_emb_avg, single_emb_min, single_emb_max = measure_embedding_latency(
        embedding_model, [sample_texts[0]]
    )

    # ── Query benchmark ──────────────────────────────────────
    print("\n[4/5] Running query benchmark...")
    test_results = []
    all_latencies = []
    category_scores = {}

    for i, (question, expected_kw, category) in enumerate(TEST_CASES):
        print(f"      [{i+1}/{len(TEST_CASES)}] {question[:50]}...")
        avg_ms, min_ms, max_ms, response = measure_latency(pipeline, question, k=3, runs=2)
        all_latencies.append(avg_ms)

        answer = response.answer if response else ""
        confidence = response.confidence if response else 0.0
        num_sources = len(response.sources) if response else 0

        hits, total_kw, matched, missed = check_keyword_hit(answer, expected_kw)
        kw_accuracy = hits / total_kw if total_kw > 0 else 0

        test_results.append({
            'question': question,
            'category': category,
            'keyword_accuracy': kw_accuracy,
            'keywords_matched': matched,
            'keywords_missed': missed,
            'confidence': confidence,
            'num_sources': num_sources,
            'avg_latency_ms': round(avg_ms, 1),
            'min_latency_ms': round(min_ms, 1),
            'max_latency_ms': round(max_ms, 1),
            'answer_length': len(answer),
        })

        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(kw_accuracy)

    # ── Aggregate stats ──────────────────────────────────────
    print("\n[5/5] Computing aggregate stats...")

    overall_kw_accuracy = statistics.mean([r['keyword_accuracy'] for r in test_results])
    full_hit_count = sum(1 for r in test_results if r['keyword_accuracy'] == 1.0)
    partial_hit_count = sum(1 for r in test_results if 0 < r['keyword_accuracy'] < 1.0)
    miss_count = sum(1 for r in test_results if r['keyword_accuracy'] == 0)

    avg_confidence = statistics.mean([r['confidence'] for r in test_results])
    avg_latency = statistics.mean(all_latencies)
    p50_latency = statistics.median(all_latencies)
    p95_latency = sorted(all_latencies)[int(len(all_latencies) * 0.95)] if len(all_latencies) >= 2 else max(all_latencies)
    min_latency = min(all_latencies)
    max_latency = max(all_latencies)

    avg_sources = statistics.mean([r['num_sources'] for r in test_results])
    avg_answer_len = statistics.mean([r['answer_length'] for r in test_results])

    # Category-level accuracy
    category_avg = {}
    for cat, scores in category_scores.items():
        category_avg[cat] = statistics.mean(scores)

    # Vector store disk size
    vs_path = RAG_DATA_DIR / "vector_store"
    vs_size_bytes = 0
    if vs_path.exists():
        for f in vs_path.iterdir():
            vs_size_bytes += f.stat().st_size
    vs_size_mb = vs_size_bytes / (1024 * 1024)

    # Cache size
    cache_path = RAG_DATA_DIR / "cache"
    cache_size_bytes = 0
    if cache_path.exists():
        for f in cache_path.iterdir():
            cache_size_bytes += f.stat().st_size
    cache_size_mb = cache_size_bytes / (1024 * 1024)

    results = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'embedding_model': embedding_model.model_name,
            'embedding_dim': embedding_dim,
            'total_chunks': num_docs,
            'indexed_vectors': num_indexed,
            'total_words': total_words,
            'avg_chunk_words': round(avg_words, 1),
            'median_chunk_words': round(median_words, 1),
            'vector_store_size_mb': round(vs_size_mb, 2),
            'cache_size_mb': round(cache_size_mb, 2),
            'language': 'Arabic (primary)',
            'retriever': 'FAISS (cosine similarity) + keyword reranking',
            'llm': 'None (extractive mode)',
        },
        'accuracy': {
            'keyword_accuracy_pct': round(overall_kw_accuracy * 100, 1),
            'exact_match_count': full_hit_count,
            'partial_match_count': partial_hit_count,
            'miss_count': miss_count,
            'total_test_cases': len(TEST_CASES),
            'avg_confidence': round(avg_confidence, 3),
        },
        'latency': {
            'avg_query_ms': round(avg_latency, 1),
            'p50_query_ms': round(p50_latency, 1),
            'p95_query_ms': round(p95_latency, 1),
            'min_query_ms': round(min_latency, 1),
            'max_query_ms': round(max_latency, 1),
            'avg_embed_single_ms': round(single_emb_avg, 1),
            'avg_embed_batch5_ms': round(emb_avg, 1),
            'system_load_ms': round(load_time, 0),
        },
        'retrieval': {
            'avg_sources_returned': round(avg_sources, 1),
            'avg_answer_length_chars': round(avg_answer_len, 0),
        },
        'category_accuracy': {cat: round(v * 100, 1) for cat, v in category_avg.items()},
        'per_query': test_results,
    }

    return results


def generate_status_md(results):
    """Generate a markdown status report."""
    s = results['system']
    a = results['accuracy']
    l = results['latency']
    r = results['retrieval']
    cat = results['category_accuracy']
    per_q = results['per_query']

    # Determine health badge
    accuracy = a['keyword_accuracy_pct']
    if accuracy >= 80:
        badge = "🟢 Healthy"
    elif accuracy >= 60:
        badge = "🟡 Fair"
    else:
        badge = "🔴 Needs Improvement"

    lines = []
    lines.append(f"# RAG Bot Status Report")
    lines.append(f"")
    lines.append(f"**Generated:** {results['timestamp'][:19].replace('T', ' ')}  ")
    lines.append(f"**Status:** {badge}  ")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## System Overview")
    lines.append(f"")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Embedding Model | `{s['embedding_model']}` |")
    lines.append(f"| Embedding Dimension | {s['embedding_dim']} |")
    lines.append(f"| Total Chunks | {s['total_chunks']} |")
    lines.append(f"| Indexed Vectors | {s['indexed_vectors']} |")
    lines.append(f"| Total Words in Corpus | {s['total_words']:,} |")
    lines.append(f"| Avg Chunk Size | {s['avg_chunk_words']} words |")
    lines.append(f"| Median Chunk Size | {s['median_chunk_words']} words |")
    lines.append(f"| Vector Store Size | {s['vector_store_size_mb']} MB |")
    lines.append(f"| Embedding Cache Size | {s['cache_size_mb']} MB |")
    lines.append(f"| Primary Language | {s['language']} |")
    lines.append(f"| Retriever | {s['retriever']} |")
    lines.append(f"| LLM Backend | {s['llm']} |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Accuracy Metrics")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| **Keyword Accuracy** | **{a['keyword_accuracy_pct']}%** |")
    lines.append(f"| Exact Matches (all keywords hit) | {a['exact_match_count']}/{a['total_test_cases']} |")
    lines.append(f"| Partial Matches | {a['partial_match_count']}/{a['total_test_cases']} |")
    lines.append(f"| Misses (no keywords hit) | {a['miss_count']}/{a['total_test_cases']} |")
    lines.append(f"| Avg Retrieval Confidence | {a['avg_confidence']} |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Latency (milliseconds)")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| **Avg Query Latency** | **{l['avg_query_ms']} ms** |")
    lines.append(f"| Median (P50) Query Latency | {l['p50_query_ms']} ms |")
    lines.append(f"| P95 Query Latency | {l['p95_query_ms']} ms |")
    lines.append(f"| Min Query Latency | {l['min_query_ms']} ms |")
    lines.append(f"| Max Query Latency | {l['max_query_ms']} ms |")
    lines.append(f"| Embedding (single query) | {l['avg_embed_single_ms']} ms |")
    lines.append(f"| Embedding (batch of 5) | {l['avg_embed_batch5_ms']} ms |")
    lines.append(f"| System Cold Start | {l['system_load_ms']} ms |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Retrieval Quality")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Avg Sources per Query | {r['avg_sources_returned']} |")
    lines.append(f"| Avg Answer Length | {int(r['avg_answer_length_chars'])} chars |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Accuracy by Category")
    lines.append(f"")
    lines.append(f"| Category | Accuracy |")
    lines.append(f"|---|---|")
    for cat_name, cat_acc in sorted(cat.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(cat_acc / 10) + "░" * (10 - int(cat_acc / 10))
        lines.append(f"| {cat_name} | {bar} {cat_acc}% |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Per-Query Breakdown")
    lines.append(f"")
    lines.append(f"| # | Question | Category | KW Acc | Confidence | Latency | Missed Keywords |")
    lines.append(f"|---|---|---|---|---|---|---|")
    for i, q in enumerate(per_q, 1):
        missed_str = ", ".join(q['keywords_missed']) if q['keywords_missed'] else "—"
        acc_icon = "✅" if q['keyword_accuracy'] == 1.0 else ("⚠️" if q['keyword_accuracy'] > 0 else "❌")
        lines.append(
            f"| {i} | {q['question'][:40]}{'…' if len(q['question'])>40 else ''} "
            f"| {q['category']} "
            f"| {acc_icon} {q['keyword_accuracy']*100:.0f}% "
            f"| {q['confidence']:.2f} "
            f"| {q['avg_latency_ms']}ms "
            f"| {missed_str} |"
        )
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Notes")
    lines.append(f"")
    lines.append(f"- **Keyword Accuracy** measures whether the retrieved answer contains expected key terms from ground-truth answers. It is not a semantic accuracy measure.")
    lines.append(f"- **Confidence** is derived from FAISS cosine similarity scores of the top-k retrieved chunks.")
    lines.append(f"- The system currently runs in **extractive mode** (no LLM). Answers are direct excerpts from the regulation document.")
    lines.append(f"- Latency measured on local machine; production latency may vary.")
    lines.append(f"- Test corpus: Faculty of Science, Helwan University academic regulations (Arabic).")
    lines.append(f"")

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        results = run_benchmark()
        md_report = generate_status_md(results)

        # Write STATUS.md
        status_path = PROJECT_DIR / "STATUS.md"
        with open(status_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"\n✓ Status report written to {status_path}")

        # Also save raw JSON
        json_path = PROJECT_DIR / "benchmark_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Raw results saved to {json_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  Keyword Accuracy : {results['accuracy']['keyword_accuracy_pct']}%")
        print(f"  Avg Query Latency: {results['latency']['avg_query_ms']} ms")
        print(f"  Avg Confidence   : {results['accuracy']['avg_confidence']}")
        print(f"  Test Cases       : {results['accuracy']['total_test_cases']}")
        print(f"  Exact Matches    : {results['accuracy']['exact_match_count']}/{results['accuracy']['total_test_cases']}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        traceback.print_exc()
        sys.exit(1)
