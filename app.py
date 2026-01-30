import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ==================== إعداد الصفحة ====================
st.set_page_config(
    page_title="مستشار أكاديمي - كلية العلوم جامعة حلوان",
    page_icon="🎓",
    layout="wide"
)

# ==================== تحميل البيانات ====================
@st.cache_resource
def load_data():
    # تحميل البيانات
    with open("rag_dataset_fixed.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [chunk['text'] for chunk in data]
    metadata = [chunk.get('metadata', {}) for chunk in data]
    
    # تحميل نموذج التضمين
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # إنشاء التضمينات
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # بناء فهرس FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    return {
        'texts': texts,
        'metadata': metadata,
        'model': model,
        'index': index,
        'data': data
    }

# ==================== دالة الاسترجاع ====================
def retrieve(query, k=3):
    query_emb = data_dict['model'].encode([query])
    distances, indices = data_dict['index'].search(np.array(query_emb).astype('float32'), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'text': data_dict['texts'][idx],
            'score': float(1 / (1 + distances[0][i])),
            'metadata': data_dict['metadata'][idx],
            'index': int(idx)
        })
    return results

# ==================== واجهة المستخدم ====================
st.title("🎓 مستشار أكاديمي ذكي")
st.markdown("**كلية العلوم - جامعة حلوان**")
st.markdown("---")

# تحميل البيانات
with st.spinner("جاري تحميل النظام..."):
    data_dict = load_data()

# قسم السؤال
st.subheader("📝 اطرح سؤالك")
query = st.text_input(
    "اكتب سؤالك حول اللائحة الأكاديمية:",
    placeholder="مثال: ما هي شروط التخرج؟ أو كم عدد الساعات المعتمدة المطلوبة؟",
    key="question_input"
)

# زر الإرسال
col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("بحث 🔍", type="primary", use_container_width=True)

st.markdown("---")

# عرض النتائج
if submit and query:
    with st.spinner("جاري البحث في اللائحة الأكاديمية..."):
        results = retrieve(query, k=3)
        
        # عرض الإجابة (سيتم تحسينها لاحقاً باستخدام LLM)
        st.subheader("💡 الإجابة")
        
        # دمج النصوص المسترجعة
        combined_context = "\n\n".join([f"المصدر {i+1}: {r['text']}" for i, r in enumerate(results)])
        
        st.info(combined_context[:1000] + "..." if len(combined_context) > 1000 else combined_context)
        
        st.markdown("---")
        
        # عرض المصادر
        st.subheader("📚 المصادر المسترجعة")
        
        for i, res in enumerate(results, 1):
            with st.expander(f"المصدر {i} - تشابه: {res['score']:.2%}"):
                st.write(res['text'])
                if res['metadata']:
                    st.caption(f"المصدر: {res['metadata'].get('source', 'غير معروف')}")

elif submit and not query:
    st.warning("⚠️ يرجى كتابة سؤال أولاً")

# قسم الأسئلة الشائعة
st.markdown("---")
st.subheader("❓ أسئلة مقترحة")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("شروط التخرج؟", use_container_width=True):
        st.session_state.question_input = "ما هي شروط التخرج من كلية العلوم؟"
        st.rerun()

with col2:
    if st.button("الساعات المعتمدة؟", use_container_width=True):
        st.session_state.question_input = "كم عدد الساعات المعتمدة المطلوبة للتخرج؟"
        st.rerun()

with col3:
    if st.button("المتطلبات المسبقة؟", use_container_width=True):
        st.session_state.question_input = "ما هي المتطلبات المسبقة لمقرر الرياضيات؟"
        st.rerun()

# معلومات عن النظام
with st.expander("ℹ️ معلومات عن النظام"):
    st.markdown("""
    - **النظام**: مستشار أكاديمي ذكي مبني على الذكاء الاصطناعي
    - **المصدر**: لائحة الساعات المعتمدة - كلية العلوم جامعة حلوان 2021
    - **الوظيفة**: البحث في اللائحة الأكاديمية والإجابة على الأسئلة
    - **ملاحظة**: النظام في مرحلة النموذج الأولي، سيتم تحسينه قريباً
    """)

# ==================== تشغيل التطبيق ====================
# لتشغيل التطبيق: 
# احفظ الكود في ملف باسم app.py
# ثم نفذ الأمر: streamlit run app.py