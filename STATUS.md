# RAG Bot Status Report

**Generated:** 2026-02-06 19:49:03  
**Status:** 🔴 Needs Improvement  

---

## System Overview

| Property | Value |
|---|---|
| Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Embedding Dimension | 384 |
| Total Chunks | 510 |
| Indexed Vectors | 510 |
| Total Words in Corpus | 101,535 |
| Avg Chunk Size | 199.1 words |
| Median Chunk Size | 200.0 words |
| Vector Store Size | 1.63 MB |
| Embedding Cache Size | 0.75 MB |
| Primary Language | Arabic (primary) |
| Retriever | FAISS (cosine similarity) + keyword reranking |
| LLM Backend | None (extractive mode) |

---

## Accuracy Metrics

| Metric | Value |
|---|---|
| **Keyword Accuracy** | **56.7%** |
| Exact Matches (all keywords hit) | 5/15 |
| Partial Matches | 6/15 |
| Misses (no keywords hit) | 4/15 |
| Avg Retrieval Confidence | 0.506 |

---

## Latency (milliseconds)

| Metric | Value |
|---|---|
| **Avg Query Latency** | **45.4 ms** |
| Median (P50) Query Latency | 26.6 ms |
| P95 Query Latency | 82.6 ms |
| Min Query Latency | 21.3 ms |
| Max Query Latency | 82.6 ms |
| Embedding (single query) | 16.4 ms |
| Embedding (batch of 5) | 43.5 ms |
| System Cold Start | 14981.0 ms |

---

## Retrieval Quality

| Metric | Value |
|---|---|
| Avg Sources per Query | 3 |
| Avg Answer Length | 1873 chars |

---

## Accuracy by Category

| Category | Accuracy |
|---|---|
| Academic Year | ██████████ 100.0% |
| GPA Calculation | ██████████ 100.0% |
| Registration | ██████████ 100.0% |
| Study Load | ██████████ 100.0% |
| Graduation Requirements | ████████░░ 83.3% |
| Summer Semester | ███████░░░ 75.0% |
| Credit Hour | ███████░░░ 75.0% |
| Honors | ██████░░░░ 66.7% |
| Study System | ███░░░░░░░ 33.3% |
| Study Levels | █░░░░░░░░░ 16.7% |
| Teaching Language | ░░░░░░░░░░ 0.0% |
| Departments | ░░░░░░░░░░ 0.0% |
| Programs | ░░░░░░░░░░ 0.0% |

---

## Per-Query Breakdown

| # | Question | Category | KW Acc | Confidence | Latency | Missed Keywords |
|---|---|---|---|---|---|---|
| 1 | ما هي شروط التخرج؟ | Graduation Requirements | ⚠️ 67% | 0.52 | 66.6ms | 138 |
| 2 | كم عدد الساعات المعتمدة المطلوبة للتخرج؟ | Graduation Requirements | ✅ 100% | 0.67 | 78.7ms | — |
| 3 | ما هي مستويات الدراسة؟ | Study Levels | ❌ 0% | 0.54 | 26.6ms | أربعة, مستوي, Freshman, Sophomore, Junior, Senior |
| 4 | ما هو المستوى الأول؟ | Study Levels | ⚠️ 33% | 0.43 | 25.4ms | صفر, Freshman |
| 5 | ما هو نظام الدراسة؟ | Study System | ⚠️ 33% | 0.47 | 22.7ms | ساعة, معتمدة |
| 6 | ما هي لغة التدريس؟ | Teaching Language | ❌ 0% | 0.46 | 23.4ms | الإنجليزية, نجليزية, الانجليزية, English |
| 7 | ما هو العام الدراسي؟ | Academic Year | ✅ 100% | 0.45 | 22.9ms | — |
| 8 | ما هو نظام الفصل الصيفي؟ | Summer Semester | ⚠️ 75% | 0.42 | 68.3ms | صيفي |
| 9 | ما هي الأقسام العلمية؟ | Departments | ❌ 0% | 0.60 | 23.0ms | الرياضيات, الفيزياء, الكيمياء, الجيولوجيا |
| 10 | كيف يتم حساب المعدل التراكمي؟ | GPA Calculation | ✅ 100% | 0.48 | 82.6ms | — |
| 11 | ما هي متطلبات التسجيل؟ | Registration | ✅ 100% | 0.45 | 63.9ms | — |
| 12 | ما هي شروط الحصول على مرتبة الشرف؟ | Honors | ⚠️ 67% | 0.41 | 67.1ms | شرف |
| 13 | ما هو العبء الدراسي؟ | Study Load | ✅ 100% | 0.58 | 21.3ms | — |
| 14 | ما هو معيار الساعة المعتمدة؟ | Credit Hour | ⚠️ 75% | 0.48 | 64.1ms | محاضرة |
| 15 | ما هي البرامج المتاحة في قسم الرياضيات؟ | Programs | ❌ 0% | 0.63 | 24.4ms | رياضيات, حاسب, إحصاء, برنامج |

---

## Notes

- **Keyword Accuracy** measures whether the retrieved answer contains expected key terms from ground-truth answers. It is not a semantic accuracy measure.
- **Confidence** is derived from FAISS cosine similarity scores of the top-k retrieved chunks.
- The system currently runs in **extractive mode** (no LLM). Answers are direct excerpts from the regulation document.
- Latency measured on local machine; production latency may vary.
- Test corpus: Faculty of Science, Helwan University academic regulations (Arabic).
