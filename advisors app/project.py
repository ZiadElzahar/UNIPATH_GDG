import pandas as pd
import matplotlib.pyplot as plt


class Student:
    def __init__(self, data_row):
        self.id = data_row['Student_ID']
        self.name = data_row['Name']
        self.payment = data_row['Payment_Status']
        self.gpa = data_row['CGPA']
        self.grades = data_row.iloc[4:-1] 

    def show_profile(self):
        print(f"\n[ Student Profile ]")
        print(f"Name: {self.name} | ID: {self.id}")
        print(f"GPA: {self.gpa} | Payment: {self.payment}")

        plt.figure(figsize=(8, 4))
        self.grades.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f"Academic Performance: {self.name}")
        plt.ylabel("Grade")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class Advisor:
    def __init__(self, adv_id, name):
        self.id = adv_id
        self.name = name
        self.my_students = []

    def add_student(self, student_obj):
        self.my_students.append(student_obj)

    def show_dashboard(self):
        print(f"\nWelcome, Dr. {self.name}!")
        print(f"You are supervising {len(self.my_students)} students.")
        
        gpas = [s.gpa for s in self.my_students]
        plt.figure(figsize=(6, 4))
        plt.hist(gpas, bins=5,color='orchid', edgecolor='black')
        plt.title(f"GPA Distribution for Dr. {self.name}'s Students")
        plt.xlabel("GPA")
        plt.ylabel("Number of Students")
        plt.show()      


class AcademicSystem:
    def __init__(self):
        self.advisors = {}
        self.all_students = {} 

    def load_data(self):
        adv_df = pd.read_csv('academic_advisors.csv')
        for _, row in adv_df.iterrows():
            self.advisors[row['Advisor_ID']] = Advisor(row['Advisor_ID'], row['Advisor_Name'])

        files = ['Year_1_Students.csv', 'Year_2_Students.csv', 'Year_3_Students.csv', 'Year_4_Students.csv']
        for file in files:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                student = Student(row)
                self.all_students[student.id] = student
                adv_id = row['Advisor_ID']
                if adv_id in self.advisors:
                    self.advisors[adv_id].add_student(student)

    def start(self):
        self.load_data()
        print("--- Academic Support System ---")
        
        try:
            input_adv = int(input("Enter Advisor ID: "))
            if input_adv in self.advisors:
                advisor = self.advisors[input_adv]
                advisor.show_dashboard()
                
                while True:
                    input_std = input("\nEnter Student ID to view details (or 'exit' to quit): ")
                    if input_std.lower() == 'exit': break
                    
                    std_id = int(input_std)
                    if std_id in self.all_students:
                        student = self.all_students[std_id]
                        student.show_profile()
                    else:
                        print("Student ID not found!")
            else:
                print("Advisor ID not found!")
        except ValueError:
            print("Please enter a valid numeric ID.")

if __name__ == "__main__":
    app = AcademicSystem()
    app.start()