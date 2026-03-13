import streamlit as st
import json
import numpy as np
import os
import requests
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

# ==================== إعداد الصفحة ====================
st.set_page_config(
    page_title="مستشار أكاديمي - كلية العلوم جامعة حلوان",
    page_icon="🎓",
    layout="wide"
)

# ==================== إعدادات Groq API (في الشريط الجانبي) ====================
groq_api_key = os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.title("⚙️ الإعدادات")

    # اختيار النموذج
    groq_model = st.selectbox(
        "اختر نموذج الذكاء الاصطناعي:",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    
    st.markdown("---")
    st.caption("النظام مبني على لائحة الساعات المعتمدة 2021")

# ==================== تحميل نظام RAG ====================
@st.cache_resource
def load_rag_system():
    repo_root = Path(__file__).resolve().parent
    # البحث عن ملف البيانات
    possible_paths = [
        str(repo_root / "data" / "rag_dataset_fixed.json"),
        str(repo_root / "rag_dataset_fixed.json"),
        os.path.join(os.getcwd(), "rag_dataset_fixed.json"),
    ]
    
    json_path = None
    for path in possible_paths:
        if os.path.exists(path):
            json_path = path
            break
    
    if json_path is None:
        st.error(f"""
        ❌ لم يتم العثور على ملف البيانات!
        
        الملف المطلوب: `rag_dataset_fixed.json`
        المجلد الحالي: `{os.getcwd()}`
        
        الحل:
        1. ضع الملف في نفس مجلد `app.py`
        2. أو غيّر مكان تشغيل التطبيق إلى المجلد الصحيح
        """)
        st.stop()
    
    # تحميل البيانات
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [chunk['text'] for chunk in data]
    metadata = [chunk.get('metadata', {}) for chunk in data]
    
    # تحميل نموذج التضمين (يدعم العربية)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
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
def retrieve_chunks(query, k=5):
    query_emb = st.session_state.rag_system['model'].encode([query])
    distances, indices = st.session_state.rag_system['index'].search(
        np.array(query_emb).astype('float32'), k
    )
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'text': st.session_state.rag_system['texts'][idx],
            'score': float(1 / (1 + distances[0][i])),
            'metadata': st.session_state.rag_system['metadata'][idx],
            'index': int(idx)
        })
    return results

# ==================== دالة توليد الإجابة عبر Groq ====================
def generate_answer_groq(query, chunks, api_key, model="llama-3.3-70b-versatile"):
    # تنسيق السياق من المقاطع المسترجعة (مع تحديد الطول لتجنب تجاوز الحد)
    context = "\n\n".join([
        f"[مصدر {i+1}] {chunk['text'][:1000]}"  # حد 1000 حرف لكل مقطع
        for i, chunk in enumerate(chunks[:5])  # أقصى 5 مقاطع
    ])
    
    # صياغة البرومبت المحسّن للعربية ولائحة جامعة حلوان
    system_prompt = """أنت مستشار أكاديمي خبير في لائحة الساعات المعتمدة لكلية العلوم بجامعة حلوان (اللائحة المعتمدة بقرار وزير التعليم العالي رقم 3257 لسنة 2021).

تعليمات هامة:
1. استند فقط على المعلومات المرجعية المقدمة
2. إذا لم تجد إجابة كافية، قل: "لم أجد معلومات كافية في اللائحة الأكاديمية الحالية"
3. كن دقيقاً وواضحاً ومختصراً
4. لا تختلق معلومات خارج السياق المقدم
5. أجب باللغة العربية"""
    
    user_prompt = f"""المعلومات المرجعية من اللائحة الأكاديمية:
{context}

السؤال: {query}

الإجابة:"""

    try:
        # استدعاء Groq API (OpenAI compatible)
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
            "top_p": 0.9
        }
        
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        # استخراج النص من الاستجابة
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "⚠️ لم يتم توليد إجابة من النموذج (استجابة فارغة)"
    
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "400" in error_msg:
            return "❌ خطأ: طلب غير صالح"
        elif "401" in error_msg or "403" in error_msg:
            return "❌ خطأ: مفتاح API غير صحيح أو منتهي الصلاحية"
        elif "429" in error_msg:
            return "⚠️ تم تجاوز حد الاستخدام. يرجى المحاولة لاحقاً."
        else:
            return f"⚠️ خطأ في الاتصال بـ Groq: {str(e)[:150]}"
    except Exception as e:
        return f"⚠️ خطأ غير متوقع: {str(e)[:150]}"

# ==================== تهيئة النظام ====================
if 'rag_system' not in st.session_state:
    with st.spinner("جاري تحميل النظام الأكاديمي..."):
        st.session_state.rag_system = load_rag_system()
        st.success(f"✓ تم تحميل {len(st.session_state.rag_system['texts'])} مقطع من لائحة 2021", icon="✅")

# ==================== واجهة المستخدم الرئيسية ====================
st.title("🎓 مستشار أكاديمي ذكي - كلية العلوم جامعة حلوان")
st.markdown("**بناءً على لائحة الساعات المعتمدة 2021 (قرار وزيري رقم 3257)**")
st.markdown("---")

# قسم السؤال
st.subheader("📝 اطرح سؤالك الأكاديمي")
query = st.text_input(
    "مثال: كم عدد الساعات المعتمدة للتخرج؟ أو ما هي شروط الالتحاق بقسم الرياضيات؟",
    placeholder="اكتب سؤالك هنا...",
    key="user_query"
)

col1, col2 = st.columns([1, 5])
with col1:
    submit_btn = st.button("ابحث وأجب 🤖", type="primary", use_container_width=True)

st.markdown("---")

# معالجة السؤال
if submit_btn and query:
    # 1. استرجاع المقاطع
    with st.spinner("جاري البحث في اللائحة الأكاديمية..."):
        chunks = retrieve_chunks(query, k=5)
    
    # 2. توليد الإجابة عبر Groq (إذا توفر المفتاح)
    if groq_api_key:
        with st.spinner("جاري توليد الإجابة عبر Groq..."):
            answer = generate_answer_groq(
                query=query,
                chunks=chunks,
                api_key=groq_api_key,
                model=groq_model
            )
    else:
        answer = None
    
    # 3. عرض النتائج
    st.subheader("💡 الإجابة")
    
    if answer and groq_api_key:
        # عرض الإجابة المولدة
        st.markdown(
            f"""
            <div style="
                background-color:#e8f4f8; 
                padding:20px; 
                border-radius:12px; 
                border-right:4px solid #2196F3;
                font-size:18px;
                line-height:1.8;
                direction: rtl;
                text-align: right;
            ">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # عرض السياق الخام كحل بديل
        st.info("لم يتم إدخال GROQ_API_KEY. إليك المعلومات ذات الصلة من اللائحة:", icon="ℹ️")
        for i, chunk in enumerate(chunks[:3], 1):
            st.markdown(f"**المصدر {i}:** {chunk['text'][:400]}...")
    
    # 4. عرض المصادر
    st.markdown("---")
    st.subheader("📚 المصادر المسترجعة من اللائحة")
    
    for i, chunk in enumerate(chunks, 1):
        score_pct = f"{chunk['score']*100:.1f}%"
        with st.expander(f"المصدر {i} (مطابقة: {score_pct})"):
            st.write(chunk['text'])
            if chunk['metadata'].get('source'):
                st.caption(f"المرجع: {chunk['metadata']['source']}")

elif submit_btn and not query:
    st.warning("⚠️ يرجى كتابة سؤال قبل البحث")
# ==================== تذييل الصفحة ====================
st.markdown("---")
st.caption("RAG_Based ChatBot Academic Advisor App © 2026 CampusBrain Team , All rights reserved." )