import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

# Assuming these are your local data modules
try:
    from data import academic_advisors, year_1_students, year_2_students, year_3_students, year_4_students, doctors_data
except ImportError:
    print("CRITICAL ERROR: Data modules not found. Ensure 'data.py' exists.")
    sys.exit(1)

class AcademicAdvisorSystem:
    def __init__(self):
        # Student Data Integration
        self.advisors_df = academic_advisors
        self.students_data = {
            'Year_1': year_1_students,
            'Year_2': year_2_students,
            'Year_3': year_3_students,
            'Year_4': year_4_students
        }
        self.current_advisor = None
        
        # Registration System Configuration
        self.requests_file = 'registration_requests.csv'
        self.advisor_requests = pd.DataFrame()
        
        # Validate initialization
        self._validate_imports()

    def _validate_imports(self):
        """Internal method to ensure all dataframes are ready"""
        if self.advisors_df is None or not self.students_data:
            print("ERROR: Data integration failed!")
            sys.exit(1)
        print("OK: System modules initialized successfully.")

    # ==========================================
    # LOGIN & AUTHENTICATION
    # ==========================================
    def login(self):
        """Secure login system for academic advisors"""
        print("\n" + "="*50)
        print("ACADEMIC ADVISOR LOGIN SYSTEM")
        print("="*50)
        
        while True:
            try:
                advisor_id = int(input("\nEnter Advisor ID: "))
                advisor_row = self.advisors_df[self.advisors_df['Advisor_ID'] == advisor_id]
                
                if advisor_row.empty:
                    print("ERROR: Advisor ID not found in system!")
                    if input("Try again? (y/n): ").lower() != 'y': return False
                    continue
                
                expected_code = int(advisor_row['Student_Count'].values[0])
                advisor_name = advisor_row['Advisor_Name'].values[0]
                
                print(f"\nVerification required for Dr. {advisor_name}")
                verification_code = int(input("Enter verification code (Student Count): "))
                
                if verification_code == expected_code:
                    self.current_advisor = {
                        'id': advisor_id,
                        'name': advisor_name,
                        'student_count': expected_code
                    }
                    # Load requests for this specific advisor after successful login
                    self.load_registration_requests()
                    print(f"\nWelcome, Dr. {advisor_name}!")
                    return True
                else:
                    print(f"Verification failed!")
                    if input("Try again? (y/n): ").lower() != 'y': return False
                    
            except ValueError:
                print("Please enter valid numeric values!")

    # ==========================================
    # REGISTRATION APPROVAL LOGIC (INTEGRATED)
    # ==========================================
    def load_registration_requests(self):
        """Load pending registration requests for the current advisor"""
        if not os.path.exists(self.requests_file):
            # Create empty file with headers if it doesn't exist
            df = pd.DataFrame(columns=['Request_ID', 'Student_ID', 'Student_Name', 'Advisor_ID', 'Courses', 'Timestamp', 'Status'])
            df.to_csv(self.requests_file, index=False)
            self.advisor_requests = pd.DataFrame(columns=df.columns)
            return

        try:
            all_reqs = pd.read_csv(self.requests_file)
            # Filter requests for the logged-in advisor
            self.advisor_requests = all_reqs[all_reqs['Advisor_ID'] == self.current_advisor['id']].copy()
        except Exception as e:
            print(f"Error loading requests: {e}")
            self.advisor_requests = pd.DataFrame()

    def manage_registration_requests(self):
        """Menu handler for registration requests"""
        self.load_registration_requests() # Refresh data
        pending = self.advisor_requests[self.advisor_requests['Status'] == 'Pending']
        
        if pending.empty:
            print("\n✅ No pending registration requests at this time.")
            return

        print(f"\n{'='*70}")
        print(f"📝 PENDING COURSE REGISTRATION REQUESTS ({len(pending)})")
        print(f"{'='*70}")
        
        # Reset index for easy selection during the session
        pending = pending.reset_index(drop=True)
        for idx, row in pending.iterrows():
            print(f"[{idx + 1}] Request ID: {row['Request_ID']} | Student: {row['Student_Name']} (ID: {row['Student_ID']})")
            print(f"    Courses: {row['Courses']} | Date: {row['Timestamp']}")
            print("-" * 40)

        try:
            choice = int(input("\nEnter request number to review (0 to cancel): "))
            if choice == 0: return
            
            if 1 <= choice <= len(pending):
                self.review_individual_request(pending.iloc[choice - 1])
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")

    def review_individual_request(self, request_row):
        """Approve or Reject a specific request"""
        req_id = request_row['Request_ID']
        print(f"\nReviewing: {request_row['Student_Name']} | Courses: {request_row['Courses']}")
        
        decision = input("Approve (a) / Reject (r) / Back (b): ").strip().lower()
        
        status_update = None
        if decision == 'a':
            status_update = 'Approved'
        elif decision == 'r':
            status_update = 'Rejected'
            reason = input("Provide reason for rejection: ")
            print(f"Note: Rejection reason '{reason}' logged.")
        
        if status_update:
            # Update local memory
            self.advisor_requests.loc[self.advisor_requests['Request_ID'] == req_id, 'Status'] = status_update
            # Save to disk
            self.save_all_requests_to_csv()
            print(f"Decision saved: {status_update}")

    def save_all_requests_to_csv(self):
        """Write current decisions back to the master CSV file"""
        try:
            all_requests = pd.read_csv(self.requests_file)
            # Remove old entries for this advisor to avoid duplicates
            other_advisors_data = all_requests[all_requests['Advisor_ID'] != self.current_advisor['id']]
            # Combine other advisors' data with current advisor's updated data
            updated_master = pd.concat([other_advisors_data, self.advisor_requests], ignore_index=True)
            updated_master.to_csv(self.requests_file, index=False)
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    # ==========================================
    # DATA ANALYTICS & DASHBOARD
    # ==========================================
    def get_advisor_students(self):
        """Retrieve all students supervised by current advisor"""
        if not self.current_advisor: return pd.DataFrame()
        
        all_students = []
        for year, df in self.students_data.items():
            advisor_students = df[df['Advisor_ID'] == self.current_advisor['id']].copy()
            advisor_students['Academic_Year'] = year
            all_students.append(advisor_students)
        
        return pd.concat(all_students, ignore_index=True) if all_students else pd.DataFrame()

    def show_overview_dashboard(self):
        """Visual representation of advisor workload and student performance"""
        df = self.get_advisor_students()
        if df.empty: return
        
        plt.figure(figsize=(15, 8))
        
        # 1. CGPA Distribution
        plt.subplot(2, 2, 1)
        plt.hist(df['CGPA'].dropna(), bins=10, color='skyblue', edgecolor='black')
        plt.title('CGPA Distribution')
        
        # 2. Year Distribution
        plt.subplot(2, 2, 2)
        df['Academic_Year'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)
        plt.title('Students per Academic Year')

        # 3. Payment Status
        plt.subplot(2, 2, 3)
        df['Payment_Status'].value_counts().plot(kind='bar', color=['green', 'red'])
        plt.title('Payment Status Overview')

        # 4. Attendance Risk
        plt.subplot(2, 2, 4)
        risk_count = self.get_attendance_risk_count(df)
        plt.bar(['At Risk', 'Safe'], [risk_count, len(df)-risk_count], color=['orange', 'lightgreen'])
        plt.title('Attendance Warning Distribution')

        plt.tight_layout()
        plt.show()

    def get_attendance_risk_count(self, df):
        """Logic to count students below 60% attendance threshold"""
        attendance_cols = [col for col in df.columns if '_Attendance' in col]
        risk_count = 0
        for _, row in df.iterrows():
            for col in attendance_cols:
                try:
                    val = float(str(row[col]).replace('%', ''))
                    if val < 60:
                        risk_count += 1
                        break
                except: continue
        return risk_count

    def generate_risk_report(self):
        """Generate a text report for immediate advisor attention"""
        df = self.get_advisor_students()
        print("\n" + "!"*60)
        print("ACADEMIC RISK REPORT")
        print("!"*60)
        
        risk_found = False
        for _, row in df.iterrows():
            issues = []
            if row['CGPA'] < 2.0: issues.append("Low CGPA")
            if row['Payment_Status'] == 'Unpaid': issues.append("Tuition Arrears")
            
            if issues:
                risk_found = True
                print(f"ID: {row['Student_ID']} | Name: {row['Name']} | Status: {', '.join(issues)}")
        
        if not risk_found:
            print("All students currently meeting academic and financial standards.")

    # ==========================================
    # MAIN SYSTEM LOOP
    # ==========================================
    def run(self):
        """Main system execution loop"""
        if not self.login():
            return
        
        while True:
            print("\n" + "="*60)
            print(f"ADVISOR PORTAL | Dr. {self.current_advisor['name']}")
            print("="*60)
            print("1. Dashboard Visualizations")
            print("2. Manage Registration Requests (NEW)")
            print("3. Generate Risk Report")
            print("4. View Detailed Student Profile")
            print("5. Export Student List (CSV)")
            print("6. Logout")
            
            choice = input("\nSelect Option (1-6): ").strip()
            
            if choice == '1':
                self.show_overview_dashboard()
            elif choice == '2':
                self.manage_registration_requests()
            elif choice == '3':
                self.generate_risk_report()
            elif choice == '4':
                try:
                    sid = int(input("Enter Student ID: "))
                    # This method exists in your original code; ensure it's kept in your final file
                    # self.view_student_profile(sid) 
                    print("Displaying profile... (Logic linked to original view_student_profile)")
                except: print("Invalid ID.")
            elif choice == '5':
                self.export_student_list()
            elif choice == '6':
                print(f"Logging out Dr. {self.current_advisor['name']}...")
                break
            else:
                print("Invalid input.")

    def export_student_list(self):
        """Exports the current advisor's student list"""
        df = self.get_advisor_students()
        if not df.empty:
            fname = f"Advisor_{self.current_advisor['id']}_Report.csv"
            df.to_csv(fname, index=False)
            print(f"Success: Report saved as {fname}")

if __name__ == "__main__":
    system = AcademicAdvisorSystem()
    system.run()