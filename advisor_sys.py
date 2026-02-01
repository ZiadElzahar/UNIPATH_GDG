import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime
from data import academic_advisors, year_1_students, year_2_students, year_3_students, year_4_students , doctors_data




class AcademicAdvisorSystem:
    def __init__(self):
        # No longer need to load from files inside the class
        self.advisors_df = academic_advisors
        self.students_data = {
            'Year_1': year_1_students,
            'Year_2': year_2_students,
            'Year_3': year_3_students,
            'Year_4': year_4_students
        }
        self.current_advisor = None
        
        # Check if data is imported successfully
        self._validate_imports()

    def _validate_imports(self):
        """Internal method to ensure all dataframes are ready"""
        if self.advisors_df is None or not self.students_data:
            print("ERROR: Data integration failed! Please check data/loaders.py")
            sys.exit(1)
        print("OK: All external data integrated successfully")
    
    def login(self):
        """Secure login system for academic advisors"""
        print("\n" + "="*50)
        print("ACADEMIC ADVISOR LOGIN SYSTEM")
        print("="*50)
        
        while True:
            try:
                advisor_id = int(input("\nEnter Advisor ID: "))
                
                # Check if advisor exists
                advisor_row = self.advisors_df[self.advisors_df['Advisor_ID'] == advisor_id]
                
                if advisor_row.empty:
                    print("ERROR: Advisor ID not found in system!")
                    choice = input("Try again? (y/n): ").lower()
                    if choice != 'y':
                        print("\nReturning to Managers Portal...")
                        return False
                    continue
                
                # Get expected verification code (student count)
                expected_code = int(advisor_row['Student_Count'].values[0])
                advisor_name = advisor_row['Advisor_Name'].values[0]
                
                # Request verification code
                print(f"\nVerification required for Dr. {advisor_name}")
                verification_code = int(input("Enter verification code: "))
                
                # Validate code
                if verification_code == expected_code:
                    self.current_advisor = {
                        'id': advisor_id,
                        'name': advisor_name,
                        'student_count': expected_code
                    }
                    print(f"\nWelcome, Dr. {advisor_name}!")
                    print(f"You are supervising {expected_code} students.")
                    return True
                else:
                    print(f"Verification failed! Expected code: {expected_code}")
                    choice = input("Try again? (y/n): ").lower()
                    if choice != 'y':
                        print("\nReturning to Managers Portal...")
                        return False
                    
            except ValueError:
                print("Please enter valid numeric values!")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user. Exiting system...")
                sys.exit(0)
    
    def get_advisor_students(self):
        """Retrieve all students supervised by current advisor"""
        if not self.current_advisor:
            return pd.DataFrame()
        
        all_students = []
        for year, df in self.students_data.items():
            advisor_students = df[df['Advisor_ID'] == self.current_advisor['id']].copy()
            advisor_students['Academic_Year'] = year
            all_students.append(advisor_students)
        
        if all_students:
            return pd.concat(all_students, ignore_index=True)
        return pd.DataFrame()
    
    def show_overview_dashboard(self):
        """Display comprehensive dashboard with multiple visualizations"""
        students_df = self.get_advisor_students()
        
        if students_df.empty:
            print("\nNo students found under your supervision!")
            return
        
        print("\n" + "="*60)
        print(f"DASHBOARD: Dr. {self.current_advisor['name']}")
        print(f"Total Students: {len(students_df)} | Academic Year: {datetime.now().year}")
        print("="*60)
        
        # 1. CGPA Distribution
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 3, 1)
        cgpa_data = students_df['CGPA'].dropna()
        plt.hist(cgpa_data, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
        plt.axvline(cgpa_data.mean(), color='red', linestyle='--', label=f'Mean: {cgpa_data.mean():.2f}')
        plt.title('CGPA Distribution', fontweight='bold')
        plt.xlabel('CGPA')
        plt.ylabel('Number of Students')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 2. Payment Status
        plt.subplot(2, 3, 2)
        payment_counts = students_df['Payment_Status'].value_counts()
        colors = ['green' if status == 'Paid' else 'red' for status in payment_counts.index]
        plt.bar(payment_counts.index, payment_counts.values, color=colors, edgecolor='black')
        plt.title('Payment Status', fontweight='bold')
        plt.ylabel('Number of Students')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # 3. Year Distribution
        plt.subplot(2, 3, 3)
        year_counts = students_df['Academic_Year'].value_counts().sort_index()
        plt.pie(year_counts.values, labels=year_counts.index, autopct='%1.1f%%', 
                colors=plt.cm.Set3(range(len(year_counts))))
        plt.title('Students by Academic Year', fontweight='bold')
        
        # 4. Grade Performance (average across courses)
        plt.subplot(2, 3, 4)
        grade_cols = [col for col in students_df.columns if '_Grade' in col and 'CGPA' not in col]
        if grade_cols:
            avg_grades = students_df[grade_cols].mean(axis=1).dropna()
            plt.hist(avg_grades, bins=12, color='coral', edgecolor='black', alpha=0.7)
            plt.title('Average Course Grades Distribution', fontweight='bold')
            plt.xlabel('Average Grade (%)')
            plt.ylabel('Number of Students')
            plt.grid(axis='y', alpha=0.3)
        
        # 5. Attendance Risk Analysis
        plt.subplot(2, 3, 5)
        attendance_cols = [col for col in students_df.columns if '_Attendance' in col]
        if attendance_cols:
            # Count students with any attendance below 60%
            at_risk = 0
            for idx, row in students_df.iterrows():
                for col in attendance_cols:
                    if pd.notna(row[col]):
                        try:
                            att_value = float(str(row[col]).replace('%', ''))
                            if np.isnan(att_value):
                                continue
                            if att_value < 60:
                                at_risk += 1
                                break
                        except:
                            continue
            
            safe = len(students_df) - at_risk
            plt.bar(['At Risk (<60%)', 'Acceptable (≥60%)'], [at_risk, safe], 
                   color=['red', 'green'], edgecolor='black')
            plt.title('Attendance Risk Analysis', fontweight='bold')
            plt.ylabel('Number of Students')
            plt.grid(axis='y', alpha=0.3)
        
        # 6. Locked Courses Analysis
        plt.subplot(2, 3, 6)
        locked_count = students_df['Locked_Courses'].apply(
            lambda x: 0 if pd.isna(x) or x == 'None' or x == 'None (Ready to Graduate)' else len(str(x).split(','))
        ).sum()
        total_courses = len(students_df) * len(grade_cols) if grade_cols else 1
        
        plt.bar(['Locked', 'Active'], [locked_count, total_courses - locked_count], 
               color=['orange', 'blue'], edgecolor='black')
        plt.title('Course Status Overview', fontweight='bold')
        plt.ylabel('Number of Courses')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Academic Dashboard - Dr. {self.current_advisor["name"]}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.show()
        
        # Print summary statistics
        self.print_summary_statistics(students_df)
    
    def print_summary_statistics(self, df):
        """Print key statistics for advisor"""
        print("\nKEY STATISTICS:")
        print("-" * 60)
        print(f"Total Students: {len(df)}")
        print(f"Average CGPA: {df['CGPA'].mean():.2f}")
        print(f"Highest CGPA: {df['CGPA'].max():.2f} (Student ID: {df.loc[df['CGPA'].idxmax(), 'Student_ID']})")
        print(f"Lowest CGPA: {df['CGPA'].min():.2f} (Student ID: {df.loc[df['CGPA'].idxmin(), 'Student_ID']})")
        
        paid = len(df[df['Payment_Status'] == 'Paid'])
        unpaid = len(df[df['Payment_Status'] == 'Unpaid'])
        print(f"\nPayment Status: {paid} Paid ({paid/len(df)*100:.1f}%) | {unpaid} Unpaid ({unpaid/len(df)*100:.1f}%)")
        
        # Count students with locked courses
        locked_students = df['Locked_Courses'].apply(
            lambda x: False if pd.isna(x) or x == 'None' or x == 'None (Ready to Graduate)' else True
        ).sum()
        print(f"Students with Locked Courses: {locked_students} ({locked_students/len(df)*100:.1f}%)")
        
        # Attendance risk
        at_risk = self.get_attendance_risk_count(df)
        print(f"Students at Attendance Risk (<60%): {at_risk} ({at_risk/len(df)*100:.1f}%)")
    
    def get_attendance_risk_count(self, df):
        """Count students with any attendance below 60%"""
        attendance_cols = [col for col in df.columns if '_Attendance' in col]
        at_risk = 0
        
        for idx, row in df.iterrows():
            for col in attendance_cols:
                if pd.notna(row[col]):
                    try:
                        att_value = float(str(row[col]).replace('%', ''))
                        if np.isnan(att_value):
                            continue
                        if att_value < 60:
                            at_risk += 1
                            break
                    except:
                        continue
        return at_risk
    
    def view_student_profile(self, student_id):
        """Display detailed profile for a specific student"""
        students_df = self.get_advisor_students()
        
        # Find student
        student = students_df[students_df['Student_ID'] == student_id]
        
        if student.empty:
            print(f"\nStudent ID {student_id} not found in your supervision list!")
            return False
        
        student = student.iloc[0]
        print("\n" + "="*60)
        print(f"STUDENT PROFILE: {student['Name']} (ID: {student_id})")
        print("="*60)
        
        # Basic info
        print(f"\nAcademic Year: {student['Academic_Year']}")
        print(f"Payment Status: {student['Payment_Status']}")
        print(f"Current CGPA: {student['CGPA']:.2f}")
        
        # Locked courses check
        locked = student['Locked_Courses']
        if pd.notna(locked) and locked not in ['None', 'None (Ready to Graduate)']:
            print(f"\nLOCKED COURSES:")
            for course in str(locked).split(','):
                print(f"   - {course.strip()}")
        else:
            print("\nNo locked courses")
        
        # Attendance analysis with warnings
        print("\nATTENDANCE ANALYSIS:")
        print("-" * 60)
        attendance_issues = []
        # Only consider attendance columns that actually have a value for this student
        attendance_cols = [col for col in student.index if '_Attendance' in col and pd.notna(student[col]) and str(student[col]).strip().lower() not in ('nan', 'none', '')]
        
        for col in attendance_cols:
            course_name = col.replace('_Attendance', '')
            att_str = str(student[col]).strip()
            
            try:
                att_value = float(att_str.replace('%', ''))
                # Skip NaN results coming from strings like 'nan' or similar
                if np.isnan(att_value):
                    print(f"{course_name}: {att_str} (Invalid format)")
                    continue

                status = "OK" if att_value >= 60 else "WARN"
                print(f"{status} {course_name}: {att_value:.1f}%")

                if att_value < 60:
                    attendance_issues.append((course_name, att_value))
            except Exception:
                print(f"{course_name}: {att_str} (Invalid format)")
        
        # Show warnings if attendance issues exist
        if attendance_issues:
            print("\n" + "!"*60)
            print("ATTENDANCE WARNING: Student has critical attendance issues!")
            print("!"*60)
            for course, att in attendance_issues:
                print(f"   - {course}: {att}% (< 60% minimum required)")
            print("\nRecommended action: Schedule immediate meeting with student")
        
        # Grade visualization
        self.plot_student_grades(student)
        
        # Grade summary
        grade_cols = [col for col in student.index if '_Grade' in col and 'CGPA' not in col]
        if grade_cols:
            grades = []
            for col in grade_cols:
                try:
                    grades.append(float(student[col]))
                except:
                    continue
            
            if grades:
                print("\nGrade Summary:")
                print(f"   Average Grade: {np.mean(grades):.1f}%")
                print(f"   Highest Grade: {max(grades):.1f}%")
                print(f"   Lowest Grade: {min(grades):.1f}%")
        
        return True
    
    def plot_student_grades(self, student):
        """Create histogram of student's grades"""
        grade_cols = [col for col in student.index if '_Grade' in col and 'CGPA' not in col]
        
        if not grade_cols:
            print("\nNo grade data available for visualization")
            return
        
        grades = []
        course_names = []
        
        for col in grade_cols:
            try:
                grade = float(student[col])
                course_name = col.replace('_Grade', '').replace('_', ' ')
                grades.append(grade)
                course_names.append(course_name)
            except:
                continue
        
        if not grades:
            print("\nNo valid grade data available for visualization")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Create color gradient based on grade values
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(grades)))
        
        bars = plt.barh(course_names, grades, color=colors, edgecolor='black')
        plt.axvline(x=60, color='red', linestyle='--', linewidth=2, label='Minimum Pass (60%)')
        plt.axvline(x=np.mean(grades), color='blue', linestyle=':', linewidth=2, 
                   label=f'Average ({np.mean(grades):.1f}%)')
        
        # Add grade values on bars
        for i, (bar, grade) in enumerate(zip(bars, grades)):
            plt.text(grade + 1, i, f'{grade:.1f}%', 
                    va='center', fontweight='bold')
        
        plt.xlabel('Grade (%)', fontweight='bold')
        plt.title(f'Academic Performance: {student["Name"]} (CGPA: {student["CGPA"]:.2f})', 
                 fontweight='bold', fontsize=14)
        plt.xlim(0, 105)
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_risk_report(self):
        """Generate comprehensive risk report for at-risk students"""
        students_df = self.get_advisor_students()
        
        if students_df.empty:
            print("\nNo students to analyze!")
            return
        
        print("\n" + "="*60)
        print("AT-RISK STUDENTS REPORT")
        print("="*60)
        
        risk_students = []
        
        for idx, row in students_df.iterrows():
            risk_factors = []
            
            # CGPA risk (<2.0)
            if row['CGPA'] < 2.0:
                risk_factors.append(f"Low CGPA ({row['CGPA']:.2f})")
            
            # Payment risk
            if row['Payment_Status'] == 'Unpaid':
                risk_factors.append("Unpaid fees")
            
            # Locked courses
            if pd.notna(row['Locked_Courses']) and row['Locked_Courses'] not in ['None', 'None (Ready to Graduate)']:
                risk_factors.append(f"Locked courses: {row['Locked_Courses']}")
            
            # Attendance risk
            attendance_cols = [col for col in row.index if '_Attendance' in col]
            for col in attendance_cols:
                if pd.notna(row[col]):
                    try:
                        att_value = float(str(row[col]).replace('%', ''))
                        if np.isnan(att_value):
                            continue
                        if att_value < 60:
                            risk_factors.append(f"Low attendance in {col.replace('_Attendance', '')} ({att_value:.0f}%)")
                            break
                    except:
                        continue
            
            if risk_factors:
                risk_students.append({
                    'ID': row['Student_ID'],
                    'Name': row['Name'],
                    'CGPA': row['CGPA'],
                    'Risks': risk_factors
                })
        
        if not risk_students:
            print("\nNo at-risk students identified. All students are performing adequately!")
            return
        
        print(f"\nFound {len(risk_students)} students requiring attention:\n")
        
        for i, student in enumerate(risk_students, 1):
            print(f"{i}. {student['Name']} (ID: {student['ID']}) | CGPA: {student['CGPA']:.2f}")
            for risk in student['Risks']:
                print(f"   - {risk}")
            print()
    
    def run(self):
        """Main system execution loop"""
        if not self.login():
            return
        
        while True:
            print("\n" + "="*60)
            print("ACADEMIC ADVISOR PORTAL")
            print("="*60)
            print("\nSelect an option:")
            print("1. View Dashboard Overview")
            print("2. View Student Profile")
            print("3. Generate Risk Report")
            print("4. Export Student List")
            print("5. Logout")
            
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    self.show_overview_dashboard()
                elif choice == '2':
                    try:
                        student_id = int(input("\nEnter Student ID: "))
                        self.view_student_profile(student_id)
                    except ValueError:
                        print("Invalid Student ID format!")
                elif choice == '3':
                    self.generate_risk_report()
                elif choice == '4':
                    self.export_student_list()
                elif choice == '5':
                    print(f"\nGoodbye, Dr. {self.current_advisor['name']}!")
                    break
                else:
                    print("Invalid choice. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\n\nSystem interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
    
    def export_student_list(self):
        """Export student list to CSV file"""
        students_df = self.get_advisor_students()
        
        if students_df.empty:
            print("\nNo students to export!")
            return
        
        filename = f"advisor_{self.current_advisor['id']}_students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_cols = ['Student_ID', 'Name', 'Academic_Year', 'Payment_Status', 'CGPA', 'Locked_Courses']
        
        try:
            students_df[export_cols].to_csv(filename, index=False)
            print(f"\nStudent list exported successfully to '{filename}'")
        except Exception as e:
            print(f"\nError exporting file: {e}")


# Run the system
if __name__ == "__main__":
    system = AcademicAdvisorSystem()
    system.run()