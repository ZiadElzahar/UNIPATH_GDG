import json
import re

def clean_text(text):
    if not text:
        return ""
    
    # --- أولاً: إصلاح مشاكل اللغة العربية ---
    
    # 1. تصحيح كلمة "الالئحة" الشائعة في الملف
    text = text.replace("الالئحة", "اللائحة")
    
    # 2. تصحيح "اال" في بداية الكلمات (مثل "االمتحان" -> "الامتحان")
    text = re.sub(r'\bاال', 'ال', text)
    
    # 3. فصل الكلمات الملتصقة التي تبدأ بـ "ال" (مثل "الفصلالدراسي" -> "الفصل الدراسي")
    # قائمة بالكلمات الشائعة التي تلتصق بما قبلها
    stems = [
        "مستوى", "فصل", "معدل", "سجل", "طالب", "كلية", "جامعة", "قسم", "برنامج", 
        "خطة", "لائحة", "مقرر", "عام", "علوم", "رياضيات", "فيزياء", "كيمياء", 
        "نبات", "حيوان", "جيولوجيا", "حاسب", "احصاء", "بيولوجي", "تدريب", 
        "مشروع", "بحث", "مقال", "جداول", "شعب", "تخصص", "دراسة", "تعليم", 
        "امتحان", "نجاح", "رسوب", "تقدير", "نقاط", "ساعات", "عبء", "حذف", 
        "إضافة", "انسحاب", "مواظبة", "تأجيل", "إنذار", "مراقبة", "انقطاع", 
        "قيد", "تطبيق", "معايير", "توصيف", "مجموعة", "محتوى", "متطلب", 
        "ارشاد", "مرشد", "نظام", "لجنة", "وكيل", "عميد", "رئيس", "عضو", 
        "هيئة", "تدريس", "معمل", "محاضرة", "تمرين", "درجة", "بكالوريوس"
    ]
    stems_pattern = "|".join(stems)
    # البحث عن حرف غير مسافة يليه "ال" ثم إحدى الكلمات، وفصلهم بمسافة
    text = re.sub(fr'([^\s])(ال(?:{stems_pattern}))', r'\1 \2', text)
    
    # 4. دمج الحروف المفصولة في نهاية الكلمات (مثل "الرسو ب" -> "الرسوب")
    # الحروف المستهدفة: ة، ى، ي، ب، ت، ء، ر، ز
    text = re.sub(r'([\u0600-\u06FF]{2,})\s+([ةىيبرزتء])\b', r'\1\2', text)

    # 5. تصحيح المسافات حول علامات الترقيم (اختياري لتحسين الشكل)
    text = re.sub(r'\s+([.,،])', r'\1', text)

    # --- ثانياً: إصلاح اللغة الإنجليزية المعكوسة ---
    
    def reverse_english_segment(match):
        content = match.group(0)
        # تقسيم النص إلى كلمات
        words = content.split()
        if len(words) > 1:
            # عكس ترتيب الكلمات (وهذا يصلح الأقواس المعكوسة تلقائياً إذا كانت ملتصقة بالكلمات)
            return " ".join(words[::-1])
        return content

    # البحث عن النصوص الإنجليزية (حروف وأرقام وعلامات ترقيم) وعكس ترتيب كلماتها
    # الشرط: يجب أن يبدأ بحرف إنجليزي وينتهي بحرف أو رقم أو قوس
    text = re.sub(r'[A-Za-z][A-Za-z0-9\s\.\-\(\),:]*[A-Za-z0-9\)]', reverse_english_segment, text)
    
    return text

def process_file(input_file, output_file):
    print(f"Loading {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Processing text...")
        for item in data:
            if 'text' in item:
                item['text'] = clean_text(item['text'])
        
        print(f"Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print("Done! File fixed successfully.")
        
    except Exception as e:
        print(f"Error: {e}")

# تشغيل الدالة
if __name__ == "__main__":
    process_file('rag_dataset.json', 'rag_dataset_fixed.json')