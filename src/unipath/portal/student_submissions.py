import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.unipath.data_access import (
    academic_advisors,
    year_1_students,
    year_2_students,
    year_3_students,
    year_4_students,
    doctors_data,
)


class StudentRegistrationSystem:
    def __init__(self):
        self.years_files = {
            1: "Year_1_Students.csv",
            2: "Year_2_Students.csv",
            3: "Year_3_Students.csv",
            4: "Year_4_Students.csv",
        }
        self.repo_root = Path(__file__).resolve().parents[3]
        self.requests_file = str(self.repo_root / "registration_requests.csv")
        self.all_students = {}
        self.load_students()

    def load_students(self):
        """Load all student data from CSV files located in the ./data directory"""
        base_dir = self.repo_root
        data_dir = base_dir / "data"

        for year, filename in self.years_files.items():
            paths_to_try = [base_dir / filename, data_dir / filename]
            df = None
            for path in paths_to_try:
                if path.exists():
                    try:
                        df = pd.read_csv(path)
                        break
                    except Exception:
                        continue

            if df is None:
                continue

            for _, row in df.iterrows():
                try:
                    student_id = int(row["Student_ID"])
                except Exception:
                    continue

                raw_locked = row.get("Locked_Courses") if year < 4 else row.get("Remaining_Courses")
                if pd.isna(raw_locked):
                    locked_val = ""
                else:
                    locked_val = str(raw_locked).strip()

                self.all_students[student_id] = {
                    "name": row["Name"],
                    "year": year,
                    "advisor_id": int(row["Advisor_ID"]),
                    "payment": row["Payment_Status"],
                    "locked_courses": locked_val,
                    "grades": row,
                }

    def generate_verification_code(self, name):
        parts = name.split()
        if len(parts) >= 2:
            return parts[0][0].lower() + parts[1][0].lower()
        return parts[0][0].lower() * 2

    def student_login(self):
        try:
            student_id = int(input("\nEnter your Student ID: ").strip())

            if student_id not in self.all_students:
                print("ERROR: Student ID not found in the system!")
                return None

            student = self.all_students[student_id]
            expected_code = self.generate_verification_code(student["name"])

            print("Verification format: First letter of first name + first letter of second name")
            print("Example: 'Mohamed Ali' -> 'ma'")

            verification = input("Enter verification code: ").strip().lower()

            if verification != expected_code:
                print(f"Verification failed! Expected code: '{expected_code}'")
                return None

            if student["year"] == 4 and "Ready to Graduate" in str(student["locked_courses"]):
                print("\nREGISTRATION ERROR: You have reached maximum credits and are ready to graduate.")
                print("You cannot register for additional courses.")
                return None

            print(f"\nWelcome, {student['name']}! (Year {student['year']})")
            return student_id, student

        except ValueError:
            print("Please enter a valid numeric ID.")
            return None
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None

    def get_available_courses(self, year):
        courses_by_year = {
            1: ["Calculus 1", "Programming 1", "Physics 1", "English 1", "Intro to CS", "Discrete Math", "Electronics"],
            2: ["Calculus 2", "Programming 2", "Data Structures", "Logic Design", "English 2", "Linear Algebra", "Probability"],
            3: ["Algorithms", "Operating Systems", "Database", "Networking", "Software Engineering", "AI", "Computer Architecture"],
            4: ["Machine Learning", "Cloud Computing", "Cybersecurity", "Graduation Project", "NLP", "Big Data", "Image Processing"],
        }
        return courses_by_year.get(year, [])

    def is_course_locked(self, student, course_name):
        locked_field = student["locked_courses"]
        course_key = str(course_name).strip().lower()

        if student["year"] == 4:
            rf = str(locked_field)
            if "none" in rf.lower() or "ready to graduate" in rf.lower():
                return True

            remaining = [c.strip().lower() for c in rf.split(",") if c.strip()]
            if not remaining:
                return True

            return course_key not in remaining

        if pd.isna(locked_field) or str(locked_field).strip().lower() in ("none", ""):
            return False

        locked_courses = [c.strip().lower() for c in str(locked_field).split(",") if c.strip()]
        return course_key in locked_courses

    def has_completed_course(self, student, course_name):
        grade_col = f"{course_name}_Grade"
        if grade_col not in student["grades"]:
            return False
        val = student["grades"][grade_col]
        if pd.isna(val):
            return False
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return False
        try:
            float(s)
            return True
        except Exception:
            return False

    def register_courses(self, student_id, student):
        print("\n" + "=" * 60)
        print("COURSE REGISTRATION")
        print("=" * 60)

        if student["year"] < 4:
            locked = student["locked_courses"]
            if pd.notna(locked) and locked not in ["None", ""]:
                print("\nCourses you CANNOT register (prerequisites not met):")
                for course in str(locked).split(","):
                    print(f"   - {course.strip()}")
            else:
                print("\nNo locked courses - you can register for any available course")
        else:
            remaining = student["locked_courses"]
            if remaining and remaining.strip() and "none" not in str(remaining).lower() and "ready to graduate" not in str(remaining).lower():
                print(f"\nCourses you can still register for: {remaining}")
            else:
                print("\nNo remaining courses available for registration.")

        if student["year"] < 4:
            target_year = student["year"] + 1
            available = self.get_available_courses(target_year)
            print(f"\nAvailable courses for registration (Year {target_year}):")
        else:
            remaining = student["locked_courses"]
            if remaining and remaining.strip() and "none" not in remaining.lower() and "ready to graduate" not in remaining.lower():
                available = [c.strip() for c in remaining.split(",") if c.strip()]
                print("\nAvailable remaining courses for registration:")
            else:
                available = []
                print("\nNo courses available for registration.")

        if not available:
            print("  (No available courses)")
        else:
            for i, course in enumerate(available, 1):
                completed = self.has_completed_course(student, course)
                locked = self.is_course_locked(student, course)
                if completed:
                    status = "COMPLETED"
                elif locked:
                    status = "LOCKED"
                else:
                    status = "OK"
                print(f"  {status} {i}. {course}")

        registered_courses = []
        print("\nEnter course numbers to register (one per line). Type 'done' when finished:")

        while True:
            choice = input(f"Course #{len(registered_courses) + 1}: ").strip().lower()

            if choice == "done":
                if not registered_courses:
                    print("No courses selected. Registration cancelled.")
                    return
                break

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    course_name = available[idx]

                    if self.has_completed_course(student, course_name):
                        print(f"   You have already completed '{course_name}'")
                        continue

                    if self.is_course_locked(student, course_name):
                        if student["year"] < 4:
                            print(f"   Cannot register: '{course_name}' - Prerequisite not met")
                        else:
                            print(f"   Cannot register: '{course_name}' - Not in your remaining courses")
                        continue

                    if course_name in registered_courses:
                        print("   Course already in your registration list.")
                        continue

                    registered_courses.append(course_name)
                    print(f"   Added: {course_name}")
                else:
                    print("   Invalid course number.")
            except ValueError:
                print("   Please enter a valid number or 'done'.")

        request_id = f"REQ{datetime.now().strftime('%Y%m%d%H%M%S')}{student_id}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        request_data = {
            "Request_ID": [request_id],
            "Student_ID": [student_id],
            "Student_Name": [student["name"]],
            "Advisor_ID": [student["advisor_id"]],
            "Courses": [";".join(registered_courses)],
            "Timestamp": [timestamp],
            "Status": ["Pending"],
        }

        df_new = pd.DataFrame(request_data)

        if os.path.exists(self.requests_file):
            df_existing = pd.read_csv(self.requests_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(self.requests_file, index=False)

        print("\n" + "=" * 60)
        print("REGISTRATION SUBMITTED SUCCESSFULLY")
        print("=" * 60)
        print(f"Request ID: {request_id}")
        print(f"Submitted to: Advisor ID {student['advisor_id']}")
        print("Courses registered:")
        for course in registered_courses:
            print(f"  - {course}")
        print("\nStatus: Pending advisor approval")
        print("You will be notified once your advisor reviews your request.")

    def run(self):
        print("\n" + "=" * 60)
        print("STUDENT COURSE REGISTRATION SYSTEM")
        print("=" * 60)

        login_result = self.student_login()
        if not login_result:
            return

        student_id, student = login_result
        self.register_courses(student_id, student)
        print("\nThank you for using the registration system. Goodbye!")


if __name__ == "__main__":
    system = StudentRegistrationSystem()
    system.run()
