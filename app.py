import streamlit as st
import json
import numpy as np
import os
import requests
from sentence_transformers import SentenceTransformer
import faiss

# ==================== إعداد الصفحة ====================
st.set_page_config(
    page_title="مستشار أكاديمي - كلية العلوم جامعة حلوان",
    page_icon="🎓",
    layout="wide"
)

# ==================== إعدادات Gemini API (في الشريط الجانبي) ====================
with st.sidebar:
    st.title("⚙️ إعدادات Gemini")
    
    # إدخال مفتاح API
    gemini_api_key = st.text_input(
        "مفتاح Google AI Studio API:",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="احصل على مفتاح من: https://aistudio.google.com/app/apikey"
    )
    #AIzaSyCT2FlqNMst9D18XYj_yhJK-bRzoJOJtpg

    # اختيار النموذج
    gemini_model = st.selectbox(
        "اختر نموذج Gemini:",
        ["gemini-2.5-flash"],
        index=0
    )
    
    st.markdown("---")
    st.caption("النظام مبني على لائحة الساعات المعتمدة 2021")

# ==================== تحميل نظام RAG ====================
@st.cache_resource
def load_rag_system():
    # البحث عن ملف البيانات
    possible_paths = [
        "rag_dataset_fixed.json",
        os.path.join(os.path.dirname(__file__), "rag_dataset_fixed.json"),
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

# ==================== دالة توليد الإجابة عبر Gemini ====================
def generate_answer_gemini(query, chunks, api_key, model="gemini-1.5-flash"):
    # تنسيق السياق من المقاطع المسترجعة (مع تحديد الطول لتجنب تجاوز الحد)
    context = "\n\n".join([
        f"[مصدر {i+1}] {chunk['text'][:1000]}"  # حد 1000 حرف لكل مقطع
        for i, chunk in enumerate(chunks[:5])  # أقصى 5 مقاطع
    ])
    
    # صياغة البرومبت المحسّن للعربية ولائحة جامعة حلوان
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

الإجابة:"""

    try:
        # استدعاء Gemini API
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800,
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
        
        # استخراج النص من الاستجابة
        if "candidates" in result and len(result["candidates"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return "⚠️ لم يتم توليد إجابة من النموذج (استجابة فارغة)"
    
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "400" in error_msg:
            return "❌ خطأ: طلب غير صالح (قد يكون المفتاح غير صحيح)"
        elif "403" in error_msg:
            return "❌ خطأ: مفتاح API غير مصرح به أو منتهي الصلاحية"
        elif "429" in error_msg:
            return "⚠️ تم تجاوز حد الاستخدام. يرجى المحاولة لاحقاً."
        else:
            return f"⚠️ خطأ في الاتصال بـ Gemini: {str(e)[:150]}"
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

# تحذير إذا لم يُدخل مفتاح API
if not gemini_api_key:
    st.warning("💡 أدخل مفتاح Gemini API في الشريط الجانبي للحصول على إجابات ذكية مولدة تلقائياً", icon="🔑")

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
    
    # 2. توليد الإجابة عبر Gemini (إذا توفر المفتاح)
    if gemini_api_key:
        with st.spinner("جاري توليد الإجابة عبر Gemini..."):
            answer = generate_answer_gemini(
                query=query,
                chunks=chunks,
                api_key=gemini_api_key,
                model=gemini_model
            )
    else:
        answer = None
    
    # 3. عرض النتائج
    st.subheader("💡 الإجابة")
    
    if answer and gemini_api_key:
        # عرض الإجابة المولدة
        # Replace current answer display with this
        st.markdown(
            f"""
                <div style="
                 background-color:#e8f4f8; 
                    padding:20px; 
                     border-radius:12px; 
                    border-left:4px solid #2196F3;
                    font-size:18px;
                    line-height:1.6;
                    min-height: 100px;
                    max-height: 80vh;
                    overflow-y: auto;
                    overflow-x: hidden;
                ">
                 {answer}
                    </div>
                    """,
                 unsafe_allow_html=True
        )
    else:
        # عرض السياق الخام كحل بديل
        st.info("لم يتم إدخال مفتاح Gemini API. إليك المعلومات ذات الصلة من اللائحة:", icon="ℹ️")
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

# ==================== أسئلة سريعة ====================
st.markdown("---")
st.subheader("⚡ أسئلة شائعة من اللائحة")
quick_questions = [
    "ما هي شروط التخرج من كلية العلوم؟",
    "كم عدد الساعات المعتمدة المطلوبة للتخرج؟",
    "ما هي مقررات المستوى الأول في قسم الرياضيات؟",
    "كيف يتم حساب المعدل التراكمي؟",
    "ما هي شروط الالتحاق بالبرنامج المزدوج؟",
    "ما هي الساعات المعتمدة لمقرر التدريب الميداني؟"
]

for q in quick_questions:
    if st.button(f"❓ {q}", use_container_width=True):
        st.session_state.user_query = q
        st.rerun()

# ==================== معلومات النظام ====================
with st.expander("ℹ️ كيفية الاستخدام"):
    st.markdown("""
    ### للحصول على أفضل النتائج:
    1. **احصل على مفتاح Gemini API** من [Google AI Studio](https://aistudio.google.com/app/apikey)
    2. **أدخل المفتاح** في الشريط الجانبي
    3. **اكتب سؤالك** بالعربية الفصحى (مثال: "شروط التخرج من قسم الكيمياء")
    4. **اضغط "ابحث وأجب"** للحصول على إجابة دقيقة من اللائحة الرسمية
    
    ### ملاحظات هامة:
    - النظام يستخدم **لائحة 2021** المعتمدة بقرار وزيري رقم 3257
    - الإجابات مبنية فقط على المعلومات الموجودة في اللائحة
    - `gemini-1.5-flash` أسرع وأرخص، و`gemini-1.5-pro` أكثر دقة
    
    > ⚠️ **تنبيه أمان**: لا تشارك مفتاح API مع الآخرين. يُنصح بحفظه كمتغير بيئة:
    > ```powershell
    > $env:GOOGLE_API_KEY="your_key_here"
    > streamlit run app.py
    > ```
    """)

# ==================== تذييل الصفحة ====================
st.markdown("---")
st.caption("نظام ذكي مبني على RAG + Gemini | كلية العلوم - جامعة حلوان © 2026")