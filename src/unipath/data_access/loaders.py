import os
from pathlib import Path

import pandas as pd

# Locate repository root from src/unipath/data_access/loaders.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data"

academic_advisors = pd.read_csv(_DATA_DIR / "academic_advisors.csv")
doctors_data = pd.read_csv(_DATA_DIR / "doctors_data.csv")
year_1_students = pd.read_csv(_DATA_DIR / "Year_1_Students.csv")
year_2_students = pd.read_csv(_DATA_DIR / "Year_2_Students.csv")
year_3_students = pd.read_csv(_DATA_DIR / "Year_3_Students.csv")
year_4_students = pd.read_csv(_DATA_DIR / "Year_4_Students.csv")
