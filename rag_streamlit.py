"""
Streamlit RAG Interface
Interactive web interface for querying the academic regulations.
"""

import streamlit as st
import os
import requests
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(
    page_title="مساعد اللوائح الأكاديمية",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
PROJECT_DIR = Path(__file__).parent
RAG_DATA_DIR = PROJECT_DIR / "rag_data"
DATA_DIR = PROJECT_DIR / "data"

# Gemini API Key (you can also set as environment variable)
GEMINI_API_KEY = "AIzaSyCT2FlqNMst9D18XYj_yhJK-bRzoJOJtpg"


def generate_answer_gemini(query: str, context: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    """Generate answer using Gemini API."""
    
    prompt = f"""أنت مستشار أكاديمي خبير في لائحة الساعات المعتمدة لكلية العلوم بجامعة حلوان (اللائحة المعتمدة بقرار وزير التعليم العالي رقم 3257 لسنة 2021).

المعلومات المرجعية من اللائحة الأكاديمية:
{context}

السؤال:
{query}

تعليمات هامة:
1. استند فقط على المعلومات المرجعية أعلاه
2. إذا لم تجد إجابة كافية، قل: "لم أجد معلومات كافية في اللائحة الأكاديمية الحالية"
3. كن دقيقاً وواضحاً ومختصراً
4. لا تختلق معلومات خارج السياق المقدم
5. اذكر رقم المادة إن وجد

الإجابة:"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000,
                "topP": 0.9
            }
        }
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "candidates" in result and len(result["candidates"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return "⚠️ لم يتم توليد إجابة من النموذج"
    
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "400" in error_msg:
            return "❌ خطأ: طلب غير صالح"
        elif "403" in error_msg:
            return "❌ خطأ: مفتاح API غير مصرح به"
        elif "429" in error_msg:
            return "⚠️ تم تجاوز حد الاستخدام. يرجى المحاولة لاحقاً."
        else:
            return f"⚠️ خطأ في الاتصال: {str(e)[:100]}"
    except Exception as e:
        return f"⚠️ خطأ: {str(e)[:100]}"


@st.cache_resource
def load_rag_pipeline():
    """Load the RAG pipeline (cached)."""
    from rag_system.embeddings import EmbeddingModel
    from rag_system.vector_store import FAISSVectorStore
    from rag_system.retriever import RAGRetriever
    from rag_system.rag_pipeline import RAGPipeline, RAGSystemBuilder
    
    vector_store_path = RAG_DATA_DIR / "vector_store"
    
    # Check if vector store exists
    if not vector_store_path.exists():
        st.warning("جاري بناء نظام RAG للمرة الأولى... هذا قد يستغرق بضع دقائق.")
        
        json_path = DATA_DIR / "rag_dataset_fixed.json"
        if not json_path.exists():
            st.error("لم يتم العثور على ملف البيانات!")
            return None
        
        builder = RAGSystemBuilder(data_dir=str(RAG_DATA_DIR))
        builder.load_from_json(str(json_path))
        pipeline = builder.build(llm_client=None, language='ar')
        return pipeline
    
    # Load existing system
    embedding_model = EmbeddingModel('multilingual-minilm')
    embedding_model.load_model()
    
    vector_store = FAISSVectorStore.load(str(vector_store_path))
    retriever = RAGRetriever(vector_store, embedding_model)
    pipeline = RAGPipeline(retriever, language='ar')
    
    return pipeline


def main():
    # Custom CSS for Arabic support
    st.markdown("""
    <style>
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        direction: rtl;
        text-align: right;
        border-left: 4px solid #1f77b4;
    }
    .confidence-high {
        color: #28a745;
    }
    .confidence-medium {
        color: #ffc107;
    }
    .confidence-low {
        color: #dc3545;
    }
    h1, h2, h3 {
        direction: rtl;
        text-align: right;
    }
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1>📚 مساعد اللوائح الأكاديمية</h1>", unsafe_allow_html=True)
    st.markdown("<p class='rtl-text'>اسأل أي سؤال عن لائحة كلية العلوم - جامعة حلوان (نظام الساعات المعتمدة)</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ إعدادات البحث")
        num_results = st.slider("عدد النتائج", min_value=1, max_value=10, value=3)
        show_sources = st.checkbox("عرض المصادر", value=True)
        show_context = st.checkbox("عرض السياق الكامل", value=False)
        
        st.markdown("---")
        st.markdown("### 💡 أمثلة على الأسئلة")
        
        example_questions = [
            "ما هي شروط التخرج؟",
            "كم عدد الساعات المعتمدة للتخرج؟",
            "ما هي مستويات الدراسة؟",
            "ما هو نظام الفصل الصيفي؟",
            "كيف يتم حساب المعدل التراكمي؟",
            "ما هي البرامج المتاحة؟",
            "ما هي شروط مرتبة الشرف؟",
            "ما هو العبء الدراسي المسموح؟"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"ex_{q}"):
                st.session_state.query = q
    
    # Load pipeline
    with st.spinner("جاري تحميل النظام..."):
        pipeline = load_rag_pipeline()
    
    if pipeline is None:
        st.error("فشل في تحميل نظام RAG. يرجى التحقق من الإعدادات.")
        return
    
    st.success("✅ النظام جاهز للاستخدام")
    
    # Query input
    query = st.text_input(
        "🔍 أدخل سؤالك هنا:",
        value=st.session_state.get('query', ''),
        placeholder="مثال: ما هي شروط التخرج من كلية العلوم؟",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔎 بحث", type="primary")
    
    # Process query
    if search_button and query:
        # Step 1: Retrieve relevant chunks
        with st.spinner("جاري البحث في اللائحة..."):
            response = pipeline.query(query, k=num_results)
        
        # Step 2: Generate answer with Gemini
        with st.spinner("جاري توليد الإجابة عبر Gemini..."):
            gemini_answer = generate_answer_gemini(
                query=query,
                context=response.context,
                api_key=GEMINI_API_KEY,
                model="gemini-2.0-flash"
            )
        
        # Display results
        st.markdown("---")
        
        # Confidence indicator
        conf = response.confidence
        if conf >= 0.7:
            conf_class = "confidence-high"
            conf_text = "عالية"
        elif conf >= 0.4:
            conf_class = "confidence-medium"
            conf_text = "متوسطة"
        else:
            conf_class = "confidence-low"
            conf_text = "منخفضة"
        
        st.markdown(f"""
        <div style='text-align: right; margin-bottom: 10px;'>
            <span>درجة مطابقة المصادر: </span>
            <span class='{conf_class}'><strong>{conf_text} ({conf:.0%})</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        # Answer from Gemini
        st.markdown("<h3>📖 الإجابة:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="
                background-color: #e8f4f8;
                padding: 20px;
                border-radius: 12px;
                border-right: 4px solid #2196F3;
                font-size: 16px;
                line-height: 1.8;
                direction: rtl;
                text-align: right;
            ">
                {gemini_answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Sources
        if show_sources and response.sources:
            st.markdown("<h3>📚 المصادر:</h3>", unsafe_allow_html=True)
            
            for i, source in enumerate(response.sources, 1):
                metadata = source.get('metadata', {})
                article = metadata.get('article_number', 'غير محدد')
                score = source.get('score', 0)
                
                with st.expander(f"المصدر {i}: مادة ({article}) - الصلة: {score:.0%}"):
                    # Show snippet of the source text
                    source_text = source.get('text', '')
                    if source_text:
                        display_text = source_text[:500]
                        if len(source_text) > 500:
                            display_text += "..."
                        st.write(display_text)
                    else:
                        st.write("لا يوجد نص متاح")
        
        # Full context (optional)
        if show_context:
            with st.expander("📄 السياق الكامل المسترجع"):
                st.text(response.context)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class='rtl-text' style='text-align: center; color: gray; font-size: 12px;'>
        نظام المساعد الأكاديمي - كلية العلوم، جامعة حلوان<br>
        يعتمد على لائحة الساعات المعتمدة 2021
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
