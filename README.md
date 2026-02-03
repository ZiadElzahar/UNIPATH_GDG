# UNIPATH 🎓

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-AI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A comprehensive university academic management system with AI-powered academic regulations assistant**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API Reference](#-api-reference)

</div>

---

## 📋 Overview

UNIPATH is a dual-purpose academic management platform designed for **Helwan University - Faculty of Science**. The system consists of two main applications:

1. **Student & Advisor Portal** (`unipath_run.py`) - Complete registration and academic advising system
2. **RAG Academic Assistant** (`rag_app.py`) - AI-powered Q&A system for academic regulations

---

## ✨ Features

### 🎓 Student Portal
- **Secure Student Login** - ID verification with name-based authentication
- **Course Registration** - Submit registration requests for courses
- **Request Management** - View, edit, and delete pending registration requests
- **Payment Status Tracking** - Check payment status before registration
- **Graduation Eligibility** - Automatic detection of graduation-ready students
- **🎉 Graduation Celebration** - Animated bubbles and confetti for Year 4 graduates
- **Year 4 Smart Logic** - Remaining courses registration, unpaid fees warning, or graduation celebration

### 👨‍🏫 Advisor Portal
- **Advisor Authentication** - Secure login with student count verification
- **Request Approval System** - Review, approve, or reject student requests with reasons
- **Student Analytics** - View assigned students with grades visualization
- **Performance Dashboard** - Side-by-side CGPA distribution and payment status charts
- **At-Risk Reports** - Generate reports for students with low GPA, attendance, or unpaid fees
- **Request Archival** - Archive and clear processed requests
- **Student Export** - Export student data to CSV with custom column selection

### 🤖 RAG Academic Assistant (Arabic/English)
- **Intelligent Q&A** - Ask questions about academic regulations in Arabic
- **Semantic Search** - FAISS-powered vector similarity search
- **Gemini AI Integration** - Google Gemini 2.0 Flash for answer generation
- **Source Citations** - View relevant article references from regulations
- **Confidence Scoring** - Reliability indicators for responses
- **Multilingual Support** - Arabic and English interface

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/UNIPATH_GDG.git
cd UNIPATH_GDG
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements_rag.txt
```

### Required Dependencies
```
# Core ML/Embedding
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
torch>=2.0.0
numpy>=1.24.0

# PDF Processing
PyMuPDF>=1.23.0

# Web Interface
streamlit>=1.28.0

# LLM Clients
openai>=1.0.0
google-generativeai>=0.3.0

# Utilities
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

---

## 💻 Usage

### Running the Student & Advisor Portal
```bash
streamlit run unipath_run.py
```
**Default URL:** `http://localhost:8501`

### Running the RAG Academic Assistant
```bash
streamlit run rag_app.py
```
**Default URL:** `http://localhost:8502`

### Running Both Applications
```bash
# Terminal 1
streamlit run unipath_run.py --server.port 8501

# Terminal 2
streamlit run rag_app.py --server.port 8502
```

---

## 🏗 Architecture

### Project Structure
```
UNIPATH_GDG/
├── 📄 unipath_run.py          # Main Student & Advisor Portal
├── 📄 rag_app.py              # RAG Academic Assistant Interface
├── 📄 advisor_sys.py          # Academic Advisor System Logic
├── 📄 student_submitions.py   # Student Registration System Logic
├── 📄 campusbrain_run.py      # CampusBrain Application
├── 📄 test_rag.py             # RAG System Tests
├── 📄 requirements_rag.txt    # Python Dependencies
├── 📄 rag_dataset_fixed.json  # Academic regulations dataset
├── 📄 registration_requests.csv # Student registration requests
│
├── 📁 data/                   # Data Files
│   ├── academic_advisors.csv  # Advisor information
│   ├── doctors_data.csv       # Faculty doctors data
│   ├── Year_1_Students.csv    # First year students
│   ├── Year_2_Students.csv    # Second year students
│   ├── Year_3_Students.csv    # Third year students
│   ├── Year_4_Students.csv    # Fourth year students
│   ├── rag_dataset_fixed.json # Academic regulations dataset
│   ├── datafix.py             # Data fixing utilities
│   └── loaders.py             # Data loading utilities
│
├── 📁 rag_system/             # RAG System Components
│   ├── __init__.py
│   ├── chunker.py             # Text chunking utilities
│   ├── embeddings.py          # Embedding model (Sentence Transformers)
│   ├── vector_store.py        # FAISS Vector Store
│   ├── retriever.py           # Document Retrieval
│   ├── llm_clients.py         # LLM API Clients
│   ├── rag_pipeline.py        # Complete RAG Pipeline
│   └── pdf_processor.py       # PDF Processing Utilities
│
└── 📁 rag_data/               # RAG Cached Data
    ├── cache/                 # Embedding Cache
    │   └── docs_510_embeddings.npy
    └── vector_store/          # FAISS Index
        ├── documents.json
        └── index.faiss
```

### System Components

#### 1. Student Registration System (`student_submitions.py`)
```python
class StudentRegistrationSystem:
    - load_students()              # Load student data from CSVs
    - student_login()              # Authenticate students
    - generate_verification_code() # Create login verification codes
    - submit_registration()        # Submit course registration
```

#### 2. Academic Advisor System (`advisor_sys.py`)
```python
class AcademicAdvisorSystem:
    - login()                      # Advisor authentication
    - load_registration_requests() # Load pending requests
    - approve_request()            # Approve student requests
    - reject_request()             # Reject student requests
    - get_advisor_students()       # Get assigned students
```

#### 3. RAG Pipeline (`rag_system/rag_pipeline.py`)
```python
class RAGPipeline:
    - query()                      # Process user questions
    - retrieve()                   # Get relevant documents
    - generate()                   # Generate AI responses

class RAGSystemBuilder:
    - load_from_json()             # Load regulations data
    - build()                      # Build complete RAG system
```

#### 4. Vector Store (`rag_system/vector_store.py`)
```python
class FAISSVectorStore:
    - add()                        # Add documents to index
    - search()                     # Similarity search
    - save() / load()              # Persistence operations
```

---

## 🔐 Authentication

### Student Login
- **Student ID**: Numeric student identifier
- **Verification Code**: First letter of first name + first letter of second name
  - Example: "Mohamed Ali" → `ma`

### Advisor Login
- **Advisor ID**: Numeric advisor identifier
- **Verification Code**: Number of assigned students (Student Count)

---

## 📊 Data Models

### Student Data Schema
| Field | Type | Description |
|-------|------|-------------|
| Student_ID | int | Unique student identifier |
| Name | str | Full student name |
| Advisor_ID | int | Assigned advisor ID |
| Payment_Status | str | Payment status (Paid/Unpaid) |
| Locked_Courses | str | Courses student must take |
| Grades | various | Academic performance data |

### Registration Request Schema
| Field | Type | Description |
|-------|------|-------------|
| Request_ID | str | Unique request identifier |
| Student_ID | int | Student's ID |
| Student_Name | str | Student's name |
| Advisor_ID | int | Advisor's ID |
| Courses | str | Requested courses |
| Timestamp | datetime | Submission time |
| Status | str | Pending/Approved/Rejected |

---

## 🤖 RAG System Details

### Embedding Models Supported
| Model Name | Identifier | Best For |
|------------|------------|----------|
| Multilingual MiniLM | `multilingual-minilm` | General Arabic/English |
| Multilingual MPNet | `multilingual-mpnet` | High accuracy |
| AraBERT | `arabic-bert` | Arabic-specific |
| LaBSE | `labse` | Best Arabic support |

### RAG Query Flow
```
User Question
     ↓
[Embedding Model] → Query Vector
     ↓
[FAISS Vector Store] → Top-K Similar Documents
     ↓
[Context Builder] → Relevant Context
     ↓
[Gemini API] → Generated Answer
     ↓
Response with Sources
```

### Configuration Options
```python
# In rag_app.py
GEMINI_API_KEY = "your-api-key"  # Or set as environment variable
MODEL = "gemini-2.0-flash"       # Gemini model version
```

---

## 🌐 API Configuration

### Gemini API Setup
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the key in `rag_app.py` or as environment variable:
```bash
export GEMINI_API_KEY="your-api-key"
```

---

## 📸 Screenshots

### Student Portal
- Course registration interface
- Request status tracking
- Payment verification
- 🎉 Graduation celebration with animated bubbles

### Advisor Portal
- Request approval dashboard with rejection reasons
- Student analytics with search functionality
- Side-by-side CGPA and payment charts
- At-risk student reports
- CSV export functionality

### RAG Assistant
- Arabic language Q&A interface
- Source citations
- Confidence indicators

---

## 🔧 Troubleshooting

### Common Issues

**1. FAISS Installation Error**
```bash
# Try CPU version
pip install faiss-cpu

# For GPU support (requires CUDA)
pip install faiss-gpu
```

**2. Sentence Transformers Loading Slow**
- First run downloads model (~500MB)
- Subsequent runs use cached model

**3. Streamlit Port Already in Use**
```bash
streamlit run unipath_run.py --server.port 8503
# or
streamlit run rag_app.py --server.port 8504
```

**4. Arabic Text Display Issues**
- Ensure browser supports RTL
- Use Chrome/Edge for best Arabic rendering

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**UNIPATH Team - GDG Project**

---

## 🙏 Acknowledgments

- Helwan University - Faculty of Science
- Google Developer Groups (GDG)
- Google Gemini AI Team
- Hugging Face for Sentence Transformers
- Facebook AI Research for FAISS

---

<div align="center">

**Made with ❤️ for Helwan University Students**

[Report Bug](https://github.com/yourusername/UNIPATH_GDG/issues) • [Request Feature](https://github.com/yourusername/UNIPATH_GDG/issues)

</div>
