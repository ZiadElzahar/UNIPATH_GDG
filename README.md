# UNIPATH GDG

UNIPATH GDG is a Python-based academic platform for university workflows and academic regulations assistance.

The repository contains:
- A Streamlit portal for student registration requests and advisor operations.
- A Retrieval-Augmented Generation (RAG) assistant for academic regulation Q&A.
- Supporting benchmark and test scripts for RAG quality and latency.

## Features

- Student portal authentication and registration request submission.
- Advisor portal authentication, request review, and student analytics.
- Risk reporting based on CGPA, attendance, and payment indicators.
- Arabic-focused RAG retrieval pipeline using semantic chunking and FAISS.
- Optional LLM integrations through pluggable client interfaces.
- Benchmark tooling for retrieval quality, confidence, and latency metrics.

## Tech Stack

- Language: Python 3.8+
- UI: Streamlit
- Data: Pandas, CSV, JSON
- Retrieval: FAISS, sentence-transformers
- Visualization: Matplotlib
- Optional LLM Providers: OpenAI, Google Gemini, Ollama, Hugging Face

## Project Structure

```text
UNIPATH_GDG/
├── apps/
│   ├── portal/app.py
│   ├── rag_cli/main.py
│   └── rag_streamlit/app.py
├── src/
│   └── unipath/
│       ├── data_access/
│       ├── portal/
│       └── rag/
├── scripts/
│   └── benchmark_rag.py
├── tests/
│   └── test_rag.py
├── data/
├── rag_data/
├── unipath_run.py
├── rag_app.py
├── campusbrain_run.py
├── requirements_rag.txt
└── README.md
```

Note:
- Root scripts remain available for backward compatibility.
- New package code lives under src/unipath.

## Installation

1. Clone repository:

```bash
git clone <your-repo-url>
cd UNIPATH_GDG
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements_rag.txt
```

## Usage

Run student/advisor Streamlit portal:

```bash
streamlit run unipath_run.py
```

Run RAG CLI assistant:

```bash
python rag_app.py
```

Run RAG Streamlit assistant:

```bash
streamlit run campusbrain_run.py
```

Run refactored entrypoints:

```bash
streamlit run apps/portal/app.py
python apps/rag_cli/main.py
streamlit run apps/rag_streamlit/app.py
```

Run tests and benchmark:

```bash
python test_rag.py
python benchmark_rag.py

# refactored wrappers
python tests/test_rag.py
python scripts/benchmark_rag.py
```

VS Code task runner:

```text
Run Task -> Portal: Streamlit
Run Task -> RAG CLI
Run Task -> RAG Streamlit
Run Task -> Run RAG Tests
Run Task -> Run Benchmark
```

Tasks are defined in `.vscode/tasks.json`.

## Environment Variables

Create a local .env file (or export in your shell) for optional provider integrations:

```env
GROQ_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

Important:
- Do not hardcode API keys in source files.
- Do not commit .env files.

## Contributing

1. Create a branch for your change.
2. Keep changes focused and atomic.
3. Run tests/benchmarks for modified components.
4. Update documentation for behavior or structure changes.
5. Open a pull request with a clear summary and validation notes.

Suggested branch naming:
- feat/<feature-name>
- fix/<bug-name>
- chore/<maintenance-task>

## License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.
Non-commercial use only. See [LICENSE](LICENSE) for full terms.
Contact: ziad.elzahar05@gmail.com
