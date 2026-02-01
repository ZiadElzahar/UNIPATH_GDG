import pandas as pd
import os

# تحديد مسار المجلد الحالي لضمان عدم حدوث خطأ في المسارات
_current_dir = os.path.dirname(__file__)

# قراءة الملفات وتحويلها إلى DataFrames
academic_advisors = pd.read_csv(os.path.join(_current_dir, 'academic_advisors.csv'))
doctors_data = pd.read_csv(os.path.join(_current_dir, 'doctors_data.csv'))
year_1_students = pd.read_csv(os.path.join(_current_dir, 'Year_1_Students.csv'))
year_2_students = pd.read_csv(os.path.join(_current_dir, 'Year_2_Students.csv'))
year_3_students = pd.read_csv(os.path.join(_current_dir, 'Year_3_Students.csv'))
year_4_students = pd.read_csv(os.path.join(_current_dir, 'Year_4_Students.csv'))