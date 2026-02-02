import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from student_submitions import StudentRegistrationSystem
from advisor_sys import AcademicAdvisorSystem

# MUST be first Streamlit command
st.set_page_config(page_title="UNIPATH", layout="wide")

# Accessibility CSS: larger fonts, bigger buttons, high-contrast
_STYLES = """
<style>
html, body, [class*="block-container"] {
  font-size: 18px !important;
}
h1 { font-size: 36px !important; }
h2 { font-size: 28px !important; }
.stButton>button { font-size: 18px !important; padding: 12px 18px; }
.stTextInput>div>div>input { font-size: 18px !important; }
.stSelectbox>div>div>div>div { font-size: 18px !important; }
.css-1x8cf1d p, .css-1x8cf1d span { font-size: 18px !important; }
</style>
"""

st.markdown(_STYLES, unsafe_allow_html=True)

# Instantiate back-end systems
student_sys = StudentRegistrationSystem()
advisor_sys = AcademicAdvisorSystem()

st.title("UNIPATH")

portal = st.sidebar.selectbox("Choose Portal", ["Student Portal", "Advisor Portal"])

# Helper functions
REQUESTS_FILE = 'registration_requests.csv'

def safe_rerun():
    """Force Streamlit rerun with compatibility for all versions."""
    try:
        st.rerun()  # Streamlit >= 1.27
    except AttributeError:
        try:
            st.experimental_rerun()  # Streamlit < 1.27
        except Exception:
            pass  # Fallback: page will refresh on next interaction

def append_request(request_row):
    """Safely append request to CSV with validation."""
    df_new = pd.DataFrame([request_row])

    # Ensure file exists with headers
    if not os.path.exists(REQUESTS_FILE):
        pd.DataFrame(columns=['Request_ID','Student_ID','Student_Name','Advisor_ID','Courses','Timestamp','Status']).to_csv(REQUESTS_FILE, index=False)

    try:
        if os.path.exists(REQUESTS_FILE):
            df_existing = pd.read_csv(REQUESTS_FILE, dtype=str)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(REQUESTS_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to save request: {e}")
        return False

def get_requests_for_student(student_id):
    """Return DataFrame of requests belonging ONLY to this student."""
    if not os.path.exists(REQUESTS_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(REQUESTS_FILE, dtype=str)
        df['Student_ID'] = df['Student_ID'].astype(str)
        # Security: Only return requests for THIS student
        return df[df['Student_ID'] == str(student_id)].copy()
    except Exception:
        return pd.DataFrame()

def get_requests_for_advisor(advisor_id):
    """Return DataFrame of requests belonging ONLY to this advisor's students."""
    if not os.path.exists(REQUESTS_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(REQUESTS_FILE, dtype=str)
        df['Advisor_ID'] = df['Advisor_ID'].astype(str)
        # Security: Only return requests for THIS advisor
        return df[df['Advisor_ID'] == str(advisor_id)].copy()
    except Exception:
        return pd.DataFrame()

def delete_request(request_id, student_id):
    """Delete a pending request ONLY if it belongs to the student and is Pending."""
    if not os.path.exists(REQUESTS_FILE):
        return False, "Requests file not found."

    try:
        df = pd.read_csv(REQUESTS_FILE, dtype=str)
        mask = (df['Request_ID'] == str(request_id)) & (df['Student_ID'] == str(student_id))

        if not mask.any():
            return False, "Request not found or you are not authorized to delete it."

        # Security: Verify student owns this request
        owner_id = df.loc[mask, 'Student_ID'].iloc[0]
        if str(owner_id) != str(student_id):
            return False, "Unauthorized: This request does not belong to you."

        status = df.loc[mask, 'Status'].iloc[0]
        if str(status).strip().lower() != 'pending':
            return False, "Cannot delete a request that is not pending."

        df = df[~mask]
        df.to_csv(REQUESTS_FILE, index=False)
        return True, f"Request {request_id} deleted successfully."
    except Exception as e:
        return False, f"Failed to delete request: {e}"

def update_request(request_id, student_id, courses):
    """Update a pending request's courses ONLY if it belongs to the student and is Pending."""
    if not os.path.exists(REQUESTS_FILE):
        return False, "Requests file not found."

    try:
        df = pd.read_csv(REQUESTS_FILE, dtype=str)
        mask = (df['Request_ID'] == str(request_id)) & (df['Student_ID'] == str(student_id))

        if not mask.any():
            return False, "Request not found or you are not authorized to edit it."

        # Security: Verify student owns this request
        owner_id = df.loc[mask, 'Student_ID'].iloc[0]
        if str(owner_id) != str(student_id):
            return False, "Unauthorized: This request does not belong to you."

        status = df.loc[mask, 'Status'].iloc[0]
        if str(status).strip().lower() != 'pending':
            return False, "Cannot edit a request that is not pending."

        df.loc[mask, 'Courses'] = str(courses)
        df.loc[mask, 'Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(REQUESTS_FILE, index=False)
        return True, f"Request {request_id} updated successfully."
    except Exception as e:
        return False, f"Failed to update request: {e}"

def approve_or_reject_request(request_id, advisor_id, decision, reason=""):
    """Approve or reject a request ONLY if it belongs to advisor's students."""
    if not os.path.exists(REQUESTS_FILE):
        return False, "Requests file not found."

    try:
        df = pd.read_csv(REQUESTS_FILE, dtype=str)
        mask = df['Request_ID'] == str(request_id)

        if not mask.any():
            return False, "Request not found."

        # Security: Verify advisor owns this request's student
        req_advisor = df.loc[mask, 'Advisor_ID'].iloc[0]
        if str(req_advisor) != str(advisor_id):
            return False, "Unauthorized: This request does not belong to your students."

        status = df.loc[mask, 'Status'].iloc[0]
        if str(status).strip().lower() != 'pending':
            return False, f"Cannot {decision} a request that is not pending."

        df.loc[mask, 'Status'] = decision.capitalize()
        if reason:
            df.loc[mask, 'Reason'] = reason
        df.to_csv(REQUESTS_FILE, index=False)
        return True, f"Request {request_id} {decision}ed successfully."
    except Exception as e:
        return False, f"Failed to {decision} request: {e}"

def archive_and_clear_requests(archive_dir='data/archives'):
    """Archive the current requests file and recreate empty one."""
    if not os.path.exists(REQUESTS_FILE):
        return False, "No requests file found."

    try:
        os.makedirs(archive_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        arc_name = os.path.join(archive_dir, f"registration_requests_archive_{ts}.csv")

        # Backup before replacing
        if os.path.exists(REQUESTS_FILE):
            import shutil
            shutil.copy2(REQUESTS_FILE, arc_name)

        # Recreate empty file
        pd.DataFrame(columns=['Request_ID','Student_ID','Student_Name','Advisor_ID','Courses','Timestamp','Status','Reason']).to_csv(REQUESTS_FILE, index=False)
        return True, f"Archived to {arc_name} and cleared requests."
    except Exception as e:
        return False, f"Failed to archive/clear requests: {e}"

# ============================================
# STUDENT PORTAL
# ============================================
if portal == "Student Portal":
    st.header("🎓 Student Portal")
    st.write("Login with your Student ID and verification code.")

    # Authentication
    if st.session_state.get('student_id') is None:
        with st.form('student_login_form'):
            sid = st.text_input("Student ID", max_chars=12, placeholder="e.g., 352300071")
            vcode = st.text_input("Verification code", max_chars=4, placeholder="e.g., 'ma' for Mohamed Ali", type="password")
            submitted = st.form_submit_button("🔐 Login")

        if submitted:
            try:
                sid_int = int(sid)
            except ValueError:
                st.error("❌ Please enter a valid numeric Student ID.")
                st.stop()

            if sid_int not in student_sys.all_students:
                st.error("❌ Student ID not found in the system!")
                st.stop()

            student = student_sys.all_students[sid_int]
            expected = student_sys.generate_verification_code(student['name'])

            if vcode.strip().lower() != expected:
                st.error(f"❌ Verification failed! Expected code: '{expected}'")
                st.stop()

            # Security: Store authenticated student
            st.session_state['student_id'] = sid_int
            st.session_state['student_name'] = student['name']
            st.session_state['student_year'] = student['year']
            st.session_state['student_advisor'] = student['advisor_id']

            st.success(f"✅ Authentication successful! Welcome, {student['name']}")
            safe_rerun()

    else:
        sid_int = st.session_state['student_id']

        # Security: Re-validate student exists
        if sid_int not in student_sys.all_students:
            st.error("❌ Session expired. Please login again.")
            st.session_state.clear()
            safe_rerun()

        student = student_sys.all_students[sid_int]

        # Security: Verify session matches current data
        if st.session_state.get('student_name') != student['name']:
            st.warning("⚠️ Session mismatch detected. Please login again.")
            st.session_state.clear()
            safe_rerun()

        st.success(f"✅ Welcome back, {student['name']} (Year {student['year']})")

        # Display student info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Advisor ID", student['advisor_id'])
        with col2:
            st.metric("Payment Status", student['payment'])
        with col3:
            st.metric("CGPA", student['grades'].get('CGPA', 'N/A'))

        # Show locked/remaining courses
        locked_info = student['locked_courses'] or '(None)'
        st.info(f"**Locked / Remaining Courses:** {locked_info}")

        # Show last request if exists
        if st.session_state.get('last_request'):
            st.success(f"✅ Last request submitted: {st.session_state.get('last_request')}")

        # Display student's requests
        s_requests = get_requests_for_student(sid_int)
        if not s_requests.empty:
            st.subheader("📝 Your Registration Requests")

            # Separate by status
            pending_reqs = s_requests[s_requests['Status'].str.strip().str.lower() == 'pending']
            approved_reqs = s_requests[s_requests['Status'].str.strip().str.lower() == 'approved']
            rejected_reqs = s_requests[s_requests['Status'].str.strip().str.lower() == 'rejected']

            if not pending_reqs.empty:
                st.markdown("### ⏳ Pending Requests")
                for idx, r in pending_reqs.iterrows():
                    rid = r.get('Request_ID', '')
                    courses = r.get('Courses', '')
                    st.markdown(f"**{rid}** — {courses}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🗑️ Delete {rid}", key=f"delete_{rid}"):
                            ok, msg = delete_request(rid, sid_int)
                            if ok:
                                st.success(msg)
                                if st.session_state.get('last_request') == rid:
                                    st.session_state.pop('last_request', None)
                                safe_rerun()
                            else:
                                st.error(msg)

                    with col2:
                        # Build available courses for editing
                        if student['year'] < 4:
                            target_year = student['year'] + 1
                            available = student_sys.get_available_courses(target_year)
                        else:
                            rf = student['locked_courses']
                            if rf and rf.strip() and 'none' not in rf.lower() and 'ready to graduate' not in rf.lower():
                                available = [c.strip() for c in rf.split(',') if c.strip()]
                            else:
                                available = []

                        # Filter allowed courses
                        allowed = [c for c in available if (not student_sys.is_course_locked(student, c)) and (not student_sys.has_completed_course(student, c))]

                        if allowed:
                            with st.expander(f"✏️ Edit {rid}"):
                                with st.form(f"edit_form_{rid}"):
                                    existing = [c.strip() for c in str(courses).split(';') if c.strip()]
                                    default_choices = [c for c in existing if c in allowed]

                                    choices = st.multiselect(
                                        "Select courses to register",
                                        options=allowed,
                                        default=default_choices,
                                        key=f"edit_multiselect_{rid}"
                                    )

                                    update = st.form_submit_button("💾 Update Request")
                                    if update:
                                        if not choices:
                                            st.warning("⚠️ Please select at least one course")
                                        else:
                                            ok, msg = update_request(rid, sid_int, ';'.join(choices))
                                            if ok:
                                                st.success(msg)
                                                st.session_state['last_request'] = rid
                                                safe_rerun()
                                            else:
                                                st.error(msg)
                        else:
                            st.info("ℹ️ No allowed courses available for editing.")

            # Show approved/rejected history
            if not approved_reqs.empty or not rejected_reqs.empty:
                with st.expander("📋 Request History"):
                    if not approved_reqs.empty:
                        st.markdown("#### ✅ Approved Requests")
                        for idx, r in approved_reqs.iterrows():
                            st.markdown(f"**{r.get('Request_ID', '')}** — {r.get('Courses', '')} (Approved)")

                    if not rejected_reqs.empty:
                        st.markdown("#### ❌ Rejected Requests")
                        for idx, r in rejected_reqs.iterrows():
                            st.markdown(f"**{r.get('Request_ID', '')}** — {r.get('Courses', '')} (Rejected)")
                            if 'Reason' in r and pd.notna(r['Reason']):
                                st.write(f"Reason: {r['Reason']}")
        else:
            st.info("ℹ️ You have no requests on file.")

        # Logout button
        if st.button("🚪 Logout"):
            st.session_state.clear()
            safe_rerun()

        # Build available course list
        if student['year'] < 4:
            target_year = student['year'] + 1
            available = student_sys.get_available_courses(target_year)
            st.subheader(f"📚 Available Courses for Registration (Year {target_year})")
        else:
            rf = student['locked_courses']
            if rf and rf.strip() and 'none' not in rf.lower() and 'ready to graduate' not in rf.lower():
                available = [c.strip() for c in rf.split(',') if c.strip()]
                st.subheader("📚 Remaining Courses Available for Registration")
            else:
                available = []
                st.info("ℹ️ No courses available for registration.")

        # Filter courses
        if available:
            allowed = [c for c in available if (not student_sys.is_course_locked(student, c)) and (not student_sys.has_completed_course(student, c))]
            locked = [c for c in available if student_sys.is_course_locked(student, c) and not student_sys.has_completed_course(student, c)]
            completed = [c for c in available if student_sys.has_completed_course(student, c)]

            if completed:
                with st.expander("✅ Completed Courses (Hidden from selection)"):
                    st.write(completed)

            if locked:
                with st.expander("🔒 Locked Courses (Prerequisites not met)"):
                    st.write(locked)

            # Check for pending requests
            pending_reqs = s_requests[s_requests['Status'].str.strip().str.lower() == 'pending'] if not s_requests.empty else pd.DataFrame()

            # Check for approved requests
            approved_reqs_block = s_requests[s_requests['Status'].str.strip().str.lower() == 'approved'] if not s_requests.empty else pd.DataFrame()

            if not approved_reqs_block.empty:
                st.success("✅ Your registration has been approved! You cannot submit another request.")
                st.markdown("### 🎉 Approved Registration")
                for idx, r in approved_reqs_block.iterrows():
                    rid = r.get('Request_ID', '')
                    courses = r.get('Courses', '')
                    timestamp = r.get('Timestamp', '')
                    st.markdown(f"**{rid}** — Courses: {courses}")
                    st.caption(f"Approved on: {timestamp}")

            elif not pending_reqs.empty:
                st.warning("⚠️ You have a pending request. Please wait for approval or delete/edit it.")

                # Show pending requests again for easy access
                for idx, r in pending_reqs.iterrows():
                    rid = r.get('Request_ID', '')
                    courses = r.get('Courses', '')
                    st.markdown(f"**{rid}** — {courses}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🗑️ Delete {rid}", key=f"pending_delete_{rid}"):
                            ok, msg = delete_request(rid, sid_int)
                            if ok:
                                st.success(msg)
                                if st.session_state.get('last_request') == rid:
                                    st.session_state.pop('last_request', None)
                                safe_rerun()
                            else:
                                st.error(msg)

                    with col2:
                        if allowed:
                            with st.expander(f"✏️ Edit {rid}"):
                                with st.form(f"pending_edit_form_{rid}"):
                                    existing = [c.strip() for c in str(courses).split(';') if c.strip()]
                                    default_choices = [c for c in existing if c in allowed]

                                    choices = st.multiselect(
                                        "Select courses",
                                        options=allowed,
                                        default=default_choices,
                                        key=f"pending_edit_{rid}"
                                    )

                                    update = st.form_submit_button("💾 Update")
                                    if update:
                                        if not choices:
                                            st.warning("⚠️ Please select at least one course")
                                        else:
                                            ok, msg = update_request(rid, sid_int, ';'.join(choices))
                                            if ok:
                                                st.success(msg)
                                                st.session_state['last_request'] = rid
                                                safe_rerun()
                                            else:
                                                st.error(msg)
                        else:
                            st.info("ℹ️ No allowed courses available for editing.")
            else:
                # No pending or accepted requests - allow new submission
                if allowed:
                    with st.form('course_request_form'):
                        st.markdown("### 📝 Submit New Registration Request")

                        choices = st.multiselect(
                            "Select courses to register",
                            options=allowed,
                            help="Choose courses you want to register for this semester"
                        )

                        submit_req = st.form_submit_button("📤 Submit Registration Request")

                        if submit_req:
                            if not choices:
                                st.warning("⚠️ Please select at least one course")
                            elif len(choices) > 7:
                                st.warning("⚠️ Maximum 7 courses per semester")
                            else:
                                # Security: Validate all courses are allowed
                                invalid = [c for c in choices if c not in allowed]
                                if invalid:
                                    st.error(f"❌ Invalid courses selected: {invalid}")
                                else:
                                    req_id = f"REQ{datetime.now().strftime('%Y%m%d%H%M%S')}{sid_int}"
                                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                                    request_row = {
                                        'Request_ID': req_id,
                                        'Student_ID': sid_int,
                                        'Student_Name': student['name'],
                                        'Advisor_ID': student['advisor_id'],
                                        'Courses': ';'.join(choices),
                                        'Timestamp': timestamp,
                                        'Status': 'Pending',
                                        'Reason': ''
                                    }

                                    ok = append_request(request_row)
                                    if ok:
                                        st.success(f"✅ Request submitted successfully! ({req_id})")
                                        st.session_state['last_request'] = req_id
                                        safe_rerun()
                                    else:
                                        st.error("❌ Failed to submit request. Please try again.")
                else:
                    st.info("ℹ️ No allowed courses available to select (all are locked or already completed).")

# ============================================
# ADVISOR PORTAL
# ============================================
elif portal == "Advisor Portal":
    st.header("👨‍🏫 Advisor Portal")
    st.write("Login with your Advisor ID and verification code (student count).")

    # Authentication
    if st.session_state.get('advisor_id') is None:
        with st.form('advisor_login_form'):
            aid = st.text_input("Advisor ID", max_chars=6, placeholder="e.g., 101")
            avcode = st.text_input("Verification code (Student Count)", type="password")
            a_login = st.form_submit_button("🔐 Login as Advisor")

        if a_login:
            try:
                aid_int = int(aid)
                avcode_int = int(avcode)
            except ValueError:
                st.error("❌ Please provide numeric Advisor ID and verification code")
                st.stop()

            # Security: Validate advisor exists
            adrow = advisor_sys.advisors_df[advisor_sys.advisors_df['Advisor_ID'] == aid_int]
            if adrow.empty:
                st.error("❌ Advisor ID not found in the system!")
                st.stop()

            expected = int(adrow['Student_Count'].values[0])
            if avcode_int != expected:
                st.error(f"❌ Verification failed! Expected code: {expected}")
                st.stop()

            # Security: Store authenticated advisor
            st.session_state['advisor_id'] = aid_int
            st.session_state['advisor_name'] = adrow['Advisor_Name'].values[0]
            st.session_state['advisor_count'] = expected

            # Set backend context
            advisor_sys.current_advisor = {
                'id': aid_int,
                'name': st.session_state['advisor_name'],
                'student_count': expected
            }

            st.success(f"✅ Logged in successfully! Welcome, Dr. {st.session_state['advisor_name']}")
            safe_rerun()

    else:
        aid_int = st.session_state['advisor_id']

        # Security: Re-validate advisor exists
        adrow = advisor_sys.advisors_df[advisor_sys.advisors_df['Advisor_ID'] == aid_int]
        if adrow.empty:
            st.error("❌ Session expired. Please login again.")
            st.session_state.clear()
            safe_rerun()

        adname = st.session_state.get('advisor_name', '')
        st.success(f"✅ Welcome back, Dr. {adname}")

        # Ensure backend context is set
        advisor_sys.current_advisor = {
            'id': aid_int,
            'name': adname,
            'student_count': st.session_state.get('advisor_count', 0)
        }

        # Logout button
        if st.button('🚪 Logout (Advisor)'):
            st.session_state.clear()
            advisor_sys.current_advisor = None
            safe_rerun()

        # Advisor actions
        option = st.selectbox(
            "Choose action",
            ["📊 Overview Dashboard", "👥 Student List", "⚠️ Generate Risk Report", "📝 Manage Requests", "📤 Export Student List"]
        )

        students_df = advisor_sys.get_advisor_students()

        if option == "📊 Overview Dashboard":
            st.subheader("Dashboard Overview")

            if students_df.empty:
                st.info("ℹ️ No students assigned to you.")
            else:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Students", len(students_df))
                with col2:
                    paid = len(students_df[students_df['Payment_Status'] == 'Paid'])
                    st.metric("Paid Students", paid)
                with col3:
                    unpaid = len(students_df[students_df['Payment_Status'] == 'Unpaid'])
                    st.metric("Unpaid Students", unpaid)
                with col4:
                    avg_gpa = students_df['CGPA'].mean()
                    st.metric("Average GPA", f"{avg_gpa:.2f}")

                # CGPA histogram
                st.markdown("### 📈 CGPA Distribution")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(students_df['CGPA'].dropna(), bins=10, color='skyblue', edgecolor='black', alpha=0.7)
                ax.set_title('CGPA Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('CGPA')
                ax.set_ylabel('Number of Students')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)

                # Payment status
                st.markdown("### 💰 Payment Status")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                payment_counts = students_df['Payment_Status'].value_counts()
                colors = ['green' if status == 'Paid' else 'red' for status in payment_counts.index]
                bars = ax2.bar(payment_counts.index, payment_counts.values, color=colors, edgecolor='black')
                ax2.set_title('Payment Status Overview', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Number of Students')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontweight='bold')

                ax2.grid(axis='y', alpha=0.3)
                st.pyplot(fig2)

                # Attendance risk
                risk_count = advisor_sys.get_attendance_risk_count(students_df)
                st.metric("⚠️ Students at Attendance Risk (<60%)", risk_count)

        elif option == "👥 Student List":
            st.subheader("Your Students")

            if students_df.empty:
                st.info("ℹ️ No students assigned to you.")
            else:
                # Display summary
                st.metric("Total Students", len(students_df))

                # Search/filter
                search_name = st.text_input("Search by name (optional)", placeholder="e.g., Mohamed")

                if search_name:
                    filtered_df = students_df[students_df['Name'].str.contains(search_name, case=False, na=False)]
                    st.write(f"Found {len(filtered_df)} student(s)")
                    display_df = filtered_df
                else:
                    display_df = students_df

                # Display students
                st.dataframe(display_df, use_container_width=True)

                # View individual student profile
                st.markdown("### 👤 View Student Profile")
                sid_view = st.text_input("Enter Student ID to view detailed profile", placeholder="e.g., 352300071")

                if sid_view:
                    try:
                        sidv = int(sid_view)
                        student_row = students_df[students_df['Student_ID'] == sidv]

                        if not student_row.empty:
                            student_data = student_row.iloc[0]

                            st.markdown(f"### 📋 Profile: {student_data['Name']} (ID: {sidv})")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Academic Year", student_data.get('Academic_Year', 'N/A'))
                                st.metric("CGPA", student_data['CGPA'])
                            with col2:
                                st.metric("Payment Status", student_data['Payment_Status'])
                                st.metric("Advisor ID", student_data['Advisor_ID'])
                            with col3:
                                # Count locked courses
                                locked = student_data.get('Locked_Courses', '')
                                if pd.notna(locked) and locked not in ['None', '']:
                                    locked_count = len([c.strip() for c in str(locked).split(',') if c.strip()])
                                    st.metric("Locked Courses", locked_count)
                                else:
                                    st.metric("Locked Courses", 0)

                            # Show attendance warnings
                            attendance_issues = []
                            for col in student_data.index:
                                if '_Attendance' in col and pd.notna(student_data[col]):
                                    try:
                                        av = float(str(student_data[col]).replace('%', ''))
                                        if av < 60:
                                            course_name = col.replace('_Attendance', '')
                                            attendance_issues.append(f"{course_name}: {av:.0f}%")
                                    except:
                                        pass

                            if attendance_issues:
                                st.warning("⚠️ **Low Attendance Courses:**")
                                for issue in attendance_issues:
                                    st.write(f"- {issue}")

                        else:
                            st.error("❌ Student not found in your group")
                    except ValueError:
                        st.error("❌ Invalid Student ID format")

        elif option == "⚠️ Generate Risk Report":
            st.subheader("⚠️ AT-RISK STUDENTS REPORT")

            if students_df.empty:
                st.info("ℹ️ No students to analyze")
            else:
                risk_list = []
                for _, r in students_df.iterrows():
                    factors = []

                    # CGPA risk
                    if r['CGPA'] < 2.0:
                        factors.append(f"🔴 Low CGPA ({r['CGPA']:.2f})")

                    # Payment risk
                    if r['Payment_Status'] == 'Unpaid':
                        factors.append("🔴 Unpaid fees")

                    # Locked courses
                    if pd.notna(r.get('Locked_Courses')) and r.get('Locked_Courses') not in ['None', '']:
                        locked_count = len([c.strip() for c in str(r.get('Locked_Courses')).split(',') if c.strip()])
                        if locked_count > 3:
                            factors.append(f"🔴 Multiple locked courses ({locked_count})")

                    # Attendance risk
                    attendance_risk = False
                    for col in r.index:
                        if '_Attendance' in col and pd.notna(r[col]):
                            try:
                                av = float(str(r[col]).replace('%', ''))
                                if av < 60:
                                    factors.append(f"🔴 Low attendance ({col.replace('_Attendance', '')}: {av:.0f}%)")
                                    attendance_risk = True
                                    break
                            except:
                                pass

                    if factors:
                        risk_list.append({
                            'ID': r['Student_ID'],
                            'Name': r['Name'],
                            'CGPA': r['CGPA'],
                            'Risks': factors,
                            'Attendance_Risk': attendance_risk
                        })

                if not risk_list:
                    st.success("✅ No at-risk students identified. All students are performing adequately!")
                else:
                    st.metric("⚠️ Total At-Risk Students", len(risk_list))

                    for s_r in risk_list:
                        with st.expander(f"⚠️ {s_r['Name']} (ID: {s_r['ID']}) — CGPA: {s_r['CGPA']:.2f}"):
                            for f in s_r['Risks']:
                                st.write(f"• {f}")

        elif option == "📝 Manage Requests":
            st.subheader("📝 Pending Registration Requests")

            if os.path.exists(REQUESTS_FILE):
                # Security: Only get requests for THIS advisor's students
                all_reqs = get_requests_for_advisor(aid_int)

                if all_reqs.empty:
                    st.info("ℹ️ No requests found for your students.")
                else:
                    # Separate by status
                    pending = all_reqs[all_reqs['Status'].str.strip().str.lower() == 'pending']
                    approved = all_reqs[all_reqs['Status'].str.strip().str.lower() == 'approved']
                    rejected = all_reqs[all_reqs['Status'].str.strip().str.lower() == 'rejected']

                    # Show pending requests
                    if not pending.empty:
                        st.markdown("### ⏳ Pending Requests")
                        st.metric("Pending Requests", len(pending))

                        for idx, row in pending.iterrows():
                            with st.container():
                                st.markdown(f"#### 📄 Request {row['Request_ID']}")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Student:** {row['Student_Name']}")
                                with col2:
                                    st.write(f"**ID:** {row['Student_ID']}")
                                with col3:
                                    st.write(f"**Date:** {row['Timestamp']}")

                                st.write(f"**Courses:** {row['Courses']}")

                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button(f"✅ Approve", key=f"approve_{row['Request_ID']}"):
                                        ok, msg = approve_or_reject_request(row['Request_ID'], aid_int, 'approved')
                                        if ok:
                                            st.success(f"✅ {msg}")
                                            safe_rerun()
                                        else:
                                            st.error(f"❌ {msg}")

                                with col_b:
                                    with st.expander(f"❌ Reject with Reason"):
                                        reason = st.text_area(
                                            "Reason for rejection",
                                            key=f"reason_{row['Request_ID']}",
                                            placeholder="e.g., Prerequisites not met, course overload, etc."
                                        )
                                        if st.button(f"❌ Reject", key=f"reject_{row['Request_ID']}"):
                                            if not reason.strip():
                                                st.warning("⚠️ Please provide a reason for rejection")
                                            else:
                                                ok, msg = approve_or_reject_request(row['Request_ID'], aid_int, 'rejected', reason)
                                                if ok:
                                                    st.success(f"❌ {msg}")
                                                    safe_rerun()
                                                else:
                                                    st.error(f"❌ {msg}")

                                st.markdown("---")

                    # Show history
                    if not approved.empty or not rejected.empty:
                        with st.expander("📋 Request History"):
                            if not approved.empty:
                                st.markdown("#### ✅ Approved Requests")
                                st.metric("Approved", len(approved))
                                st.dataframe(approved[['Request_ID', 'Student_Name', 'Courses', 'Timestamp']])

                            if not rejected.empty:
                                st.markdown("#### ❌ Rejected Requests")
                                st.metric("Rejected", len(rejected))
                                st.dataframe(rejected[['Request_ID', 'Student_Name', 'Courses', 'Timestamp', 'Reason']])
                    # ADD THE ADMIN CONTROL PANEL HERE
            else:
                st.info('ℹ️ No requests found for your students')
        # ============================================
        elif option == "📤 Export Student List":
            st.subheader("📤 Export Student Data")

            if students_df.empty:
                st.info("ℹ️ No students to export")
            else:
                st.metric("Students to Export", len(students_df))

                # Select columns to export
                all_cols = list(students_df.columns)
                default_cols = ['Student_ID', 'Name', 'Academic_Year', 'Payment_Status', 'CGPA', 'Advisor_ID']
                selected_cols = st.multiselect(
                    "Select columns to export",
                    options=all_cols,
                    default=[c for c in default_cols if c in all_cols]
                )

                if selected_cols:
                    export_df = students_df[selected_cols]

                    # Export button
                    csv = export_df.to_csv(index=False)
                    fname = f"advisor_{aid_int}_students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv,
                        file_name=fname,
                        mime="text/csv"
                    )

                    st.success(f"✅ File ready for download: `{fname}`")
                else:
                    st.warning("⚠️ Please select at least one column to export")

        # System maintenance panel (admin only)
        with st.expander("🛠️ System Maintenance (Admin Only)"):
            st.warning("⚠️ These actions affect all users")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("🧹 Clear Data Cache"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("✅ Caches cleared")
                    safe_rerun()

            with col2:
                if st.button("👤 Clear All Sessions"):
                    st.session_state.clear()
                    st.success("✅ All user sessions cleared")
                    safe_rerun()

            with col3:
                confirm_archive = st.checkbox("Confirm archive", key="confirm_archive")
                if confirm_archive and st.button("🗃️ Archive Requests"):
                    ok, msg = archive_and_clear_requests()
                    if ok:
                        st.success(msg)
                        safe_rerun()
                    else:
                        st.error(msg)

            with col4:
                confirm_delete = st.checkbox("Confirm delete ALL", key="confirm_delete_all")
                if confirm_delete and st.button("🗑️ Delete All Requests"):
                    try:
                        if os.path.exists(REQUESTS_FILE):
                            os.remove(REQUESTS_FILE)
                            pd.DataFrame(columns=['Request_ID','Student_ID','Student_Name','Advisor_ID','Courses','Timestamp','Status','Reason']).to_csv(REQUESTS_FILE, index=False)
                            st.success("✅ All requests deleted and file reset")
                            safe_rerun()
                        else:
                            st.info("No requests file to delete")
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")

# Footer
st.markdown("---")
st.markdown("**© 2026 UNIPATH University Registration System**")
st.caption("🔒 Secure • Accessible • Student-Focused")
