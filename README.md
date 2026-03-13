<div align="center">

# 🎓 UNIPATH GDG

**An intelligent academic platform for university workflows & Arabic regulation Q&A**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0078D7?style=for-the-badge)](https://faiss.ai/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

---

## 📖 About

**UNIPATH GDG** is a Python-based academic platform built for Helwan University that bridges the gap between students, advisors, and university regulations — all in one place.

| Component | Description |
|-----------|-------------|
| 🏫 **Student Portal** | Login, browse courses, and submit registration requests to your advisor |
| 🧑‍🏫 **Advisor Portal** | Review & approve/reject student requests, visualize student analytics |
| 🤖 **CampusBrain RAG** | Arabic-first AI assistant for academic regulation Q&A powered by FAISS + LLMs |

---

## ✨ Features

- 🔐 **Secure authentication** for both students and academic advisors
- 📋 **Course registration workflow** with advisor approval pipeline
- 📊 **Risk reporting** based on CGPA, attendance, and payment status
- 🧠 **Semantic RAG pipeline** with Arabic-optimized chunking and FAISS vector search
- 🔌 **Pluggable LLM backends** — OpenAI, Gemini, Groq, Ollama, Hugging Face
- 📈 **Benchmark tooling** for retrieval quality, confidence, and latency

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| UI | Streamlit |
| Data | Pandas, CSV, JSON |
| Vector Search | FAISS |
| Embeddings | sentence-transformers |
| Visualization | Matplotlib |
| LLM Providers | OpenAI · Google Gemini · Groq · Ollama · Hugging Face |

---

## 📁 Project Structure

```text
UNIPATH_GDG/
├── 📦 apps/
│   ├── portal/app.py          # Student & Advisor portal entrypoint
│   ├── rag_cli/main.py        # RAG CLI assistant
│   └── rag_streamlit/app.py   # CampusBrain Streamlit UI
│
├── 🧩 src/unipath/
│   ├── data_access/           # Centralized CSV loaders
│   ├── portal/                # Student & Advisor business logic
│   └── rag/                   # Full RAG pipeline (chunker, embeddings, FAISS, LLM)
│
├── 🧪 tests/
│   └── test_rag.py
├── ⚙️  scripts/
│   └── benchmark_rag.py
├── 📂 data/                   # Student & advisor CSV datasets
├── 🗄️  rag_data/               # FAISS index & embedding cache
├── unipath_run.py             # Legacy portal entrypoint
├── campusbrain_run.py         # Legacy RAG Streamlit entrypoint
├── rag_app.py                 # Legacy RAG CLI entrypoint
└── requirements_rag.txt
```

> **Note:** Root scripts remain for backward compatibility. Canonical code lives under `src/unipath/`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd UNIPATH_GDG
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements_rag.txt
```

---

## ▶️ Usage

### 🏫 Student / Advisor Portal

```bash
streamlit run unipath_run.py
# or
streamlit run apps/portal/app.py
```

### 🤖 CampusBrain RAG (Streamlit)

```bash
streamlit run campusbrain_run.py
# or
streamlit run apps/rag_streamlit/app.py
```

### 💻 RAG CLI

```bash
python rag_app.py
# or
python apps/rag_cli/main.py
```

### 🧪 Tests & Benchmarks

```bash
python tests/test_rag.py
python scripts/benchmark_rag.py
```

### ⚡ VS Code Task Runner

Open the Command Palette → **Tasks: Run Task** and choose:

| Task | Description |
|------|-------------|
| `Portal: Streamlit` | Launch the student/advisor portal |
| `RAG Streamlit` | Launch CampusBrain UI |
| `RAG CLI` | Run the CLI assistant |
| `Run RAG Tests` | Execute the test suite |
| `Run Benchmark` | Run RAG benchmarks |

> Tasks are defined in `.vscode/tasks.json`.

---

## 🔑 Environment Variables

Create a `.env` file (or export in your shell) before running the RAG apps:

```env
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

> ⚠️ **Never hardcode API keys in source files. Never commit `.env` to version control.**

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. 🍴 Fork the repository and create a new branch
2. 🎯 Keep changes focused and atomic
3. 🧪 Run tests/benchmarks for modified components
4. 📝 Update docs for behavior or structure changes
5. 🔃 Open a pull request with a clear summary

**Branch naming convention:**

```
feat/<feature-name>
fix/<bug-description>
chore/<maintenance-task>
```

---

## 📄 License

This project is licensed under the **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** License —
free to share and adapt for **non-commercial use only**.

See [LICENSE](LICENSE) for full terms.

📧 Contact: [ziad.elzahar05@gmail.com](mailto:ziad.elzahar05@gmail.com)

---

<div align="center">
  <sub>Built with ❤️ by the UNIPATH GDG Team · Helwan University · 2026</sub>
</div>
