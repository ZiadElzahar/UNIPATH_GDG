import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from student_submitions import StudentRegistrationSystem
from advisor_sys import AcademicAdvisorSystem

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

st.set_page_config(page_title="UNIPATH", layout="wide")

# Instantiate back-end systems
student_sys = StudentRegistrationSystem()
advisor_sys = AcademicAdvisorSystem()

st.title("UNIPATH")
#st.write("Large-font, accessible UI for course registration and advisor dashboards.")

portal = st.sidebar.selectbox("Choose Portal", ["Student Portal", "Advisor Portal"]) 

# Helper functions
REQUESTS_FILE = 'registration_requests.csv'


def safe_rerun():
    """Attempt to force a Streamlit rerun.

    Uses st.experimental_rerun() when available; otherwise toggles a session_state
    counter to trigger a rerun. This provides compatibility across Streamlit versions.
    """
    try:
        st.experimental_rerun()
    except Exception:
        # Fallback: modifying session_state triggers a rerun in Streamlit
        st.session_state["_rerun_counter"] = st.session_state.get("_rerun_counter", 0) + 1


def append_request(request_row):
    df_new = pd.DataFrame([request_row])
    try:
        if st.session_state.get('requests_file_checked') is None:
            # Ensure file exists with headers
            if not os.path.exists(REQUESTS_FILE):
                pd.DataFrame(columns=['Request_ID','Student_ID','Student_Name','Advisor_ID','Courses','Timestamp','Status']).to_csv(REQUESTS_FILE, index=False)
            st.session_state['requests_file_checked'] = True
    except Exception:
        pass

    try:
        if os.path.exists(REQUESTS_FILE):
            df_existing = pd.read_csv(REQUESTS_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(REQUESTS_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to save request: {e}")
        return False


def get_requests_for_student(student_id):
    """Return DataFrame of requests belonging to a student (may be empty)."""
    if not os.path.exists(REQUESTS_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(REQUESTS_FILE, dtype=str)
        df['Student_ID'] = df['Student_ID'].astype(str)
        return df[df['Student_ID'] == str(student_id)].copy()
    except Exception:
        return pd.DataFrame()


def delete_request(request_id, student_id):
    """Delete a pending request if it belongs to the student. Returns (ok, message)."""
    if not os.path.exists(REQUESTS_FILE):
        return False, "Requests file not found."
    try:
        df = pd.read_csv(REQUESTS_FILE, dtype=str)
        mask = (df['Request_ID'] == str(request_id)) & (df['Student_ID'] == str(student_id))
        if not mask.any():
            return False, "Request not found or you are not authorized to delete it."
        status = df.loc[mask, 'Status'].iloc[0]
        if str(status).strip().lower() != 'pending':
            return False, "Cannot delete a request that is not pending."
        df2 = df[~mask]
        df2.to_csv(REQUESTS_FILE, index=False)
        return True, f"Request {request_id} deleted."
    except Exception as e:
        return False, f"Failed to delete request: {e}"


def update_request(request_id, student_id, courses):
    """Update a pending request's courses (only if it belongs to the student and is Pending). Returns (ok, message)."""
    if not os.path.exists(REQUESTS_FILE):
        return False, "Requests file not found."
    try:
        df = pd.read_csv(REQUESTS_FILE, dtype=str)
        mask = (df['Request_ID'] == str(request_id)) & (df['Student_ID'] == str(student_id))
        if not mask.any():
            return False, "Request not found or you are not authorized to edit it."
        status = df.loc[mask, 'Status'].iloc[0]
        if str(status).strip().lower() != 'pending':
            return False, "Cannot edit a request that is not pending."
        df.loc[mask, 'Courses'] = str(courses)
        df.loc[mask, 'Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(REQUESTS_FILE, index=False)
        return True, f"Request {request_id} updated."
    except Exception as e:
        return False, f"Failed to update request: {e}"


def archive_and_clear_requests(archive_dir='data/archives'):
    """Archive the current requests file (move it into archvie folder with timestamp), then recreate an empty requests file.
    Returns (ok, message)."""
    if not os.path.exists(REQUESTS_FILE):
        return False, "No requests file found."
    try:
        os.makedirs(archive_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        arc_name = os.path.join(archive_dir, f"registration_requests_archive_{ts}.csv")
        # Move the file to the archive name
        os.replace(REQUESTS_FILE, arc_name)
        # Recreate an empty requests file with headers
        pd.DataFrame(columns=['Request_ID','Student_ID','Student_Name','Advisor_ID','Courses','Timestamp','Status']).to_csv(REQUESTS_FILE, index=False)
        return True, f"Archived to {arc_name} and cleared requests."
    except Exception as e:
        return False, f"Failed to archive/clear requests: {e}"

import os

if portal == "Student Portal":
    st.header("Student Portal")
    st.write("Login with your Student ID and verification code.")
    # If user already logged in during this session, keep them authenticated
    if st.session_state.get('student_id') is None:
        with st.form('student_login_form'):
            sid = st.text_input("Student ID", max_chars=12)
            vcode = st.text_input("Verification code", max_chars=4)
            submitted = st.form_submit_button("Login")

        if submitted:
            try:
                sid_int = int(sid)
            except Exception:
                st.error("Please enter a valid numeric Student ID.")
                st.stop()

            if sid_int not in student_sys.all_students:
                st.error("Student ID not found")
                st.stop()

            student = student_sys.all_students[sid_int]
            expected = student_sys.generate_verification_code(student['name'])
            if vcode.strip().lower() != expected:
                st.error("Verification failed")
                st.stop()

            # store in session
            st.session_state['student_id'] = sid_int
            st.success("Authentication successful. You are now logged in.")
            safe_rerun()

    else:
        sid_int = st.session_state['student_id']
        if sid_int not in student_sys.all_students:
            st.error("Stored student not found. Please login again.")
            st.session_state.pop('student_id', None)
            safe_rerun()

        student = student_sys.all_students[sid_int]
        st.success(f"Welcome back {student['name']} (Year {student['year']})")
        st.write("__Current data:__")
        st.write({
            'Advisor ID': student['advisor_id'],
            'Payment Status': student['payment'],
            'Locked / Remaining': student['locked_courses'] or '(None)',
            'CGPA (if available)': student['grades'].get('CGPA', '')
        })

        # Show last request info if recently submitted
        if st.session_state.get('last_request'):
            st.success(f"Last request submitted: {st.session_state.get('last_request')}")

        # Show student's requests and allow deletion of pending ones
        s_requests = get_requests_for_student(sid_int)
        if not s_requests.empty:
            st.subheader("Your registration requests")
            for idx, r in s_requests.iterrows():
                rid = r.get('Request_ID', '')
                courses = r.get('Courses', '')
                status = r.get('Status', '')
                st.markdown(f"**{rid}** — {courses}  \n_Status: {status}_")
                if str(status).strip().lower() == 'pending':
                    # allow deletion by student owner only
                    if st.button(f"Delete request {rid}", key=f"delete_{rid}"):
                        ok, msg = delete_request(rid, sid_int)
                        if ok:
                            st.success(msg)
                            if st.session_state.get('last_request') == rid:
                                st.session_state.pop('last_request', None)
                            safe_rerun()
                        else:
                            st.warning(msg)
                else:
                    st.info("Cannot delete request: not pending.")
        else:
            st.info("You have no requests on file.")

        if st.button("Logout"):
            st.session_state.pop('student_id', None)
            safe_rerun()

        # Build available course list (next-year for years 1-3, explicit remaining for year 4)
        if student['year'] < 4:
            target_year = student['year'] + 1
            available = student_sys.get_available_courses(target_year)
            st.subheader(f"Available courses for registration (Year {target_year})")
        else:
            rf = student['locked_courses']
            if rf and rf.strip() and 'none' not in rf.lower() and 'ready to graduate' not in rf.lower():
                available = [c.strip() for c in rf.split(',') if c.strip()]
                st.subheader("Remaining courses available for registration")
            else:
                available = []
                st.info("No courses available for registration.")

        if available:
            # Filter to only allowed courses (not locked and not completed)
            allowed = [c for c in available if (not student_sys.is_course_locked(student, c)) and (not student_sys.has_completed_course(student, c))]
            locked = [c for c in available if student_sys.is_course_locked(student, c) and not student_sys.has_completed_course(student, c)]
            completed = [c for c in available if student_sys.has_completed_course(student, c)]

            if completed:
                st.markdown("**Completed courses (hidden from selection):**")
                st.write(completed)
            if locked:
                st.markdown("**Locked / Not allowed:**")
                st.write(locked)

            # If the student has a pending request, do not allow creating a new one
            pending_df = get_requests_for_student(sid_int)
            pending = pending_df[pending_df['Status'].str.strip().str.lower() == 'pending'] if not pending_df.empty else pd.DataFrame()

            if not pending.empty:
                st.warning("You have a pending request. You cannot submit a new request until you delete or edit your pending request.")
                for idx, r in pending.iterrows():
                    rid = r.get('Request_ID', '')
                    courses = r.get('Courses', '')
                    status = r.get('Status', '')
                    st.markdown(f"**{rid}** — {courses}  \n_Status: {status}_")

                    col1, col2 = st.columns([1,2])
                    with col1:
                        if st.button(f"Delete request {rid}", key=f"student_delete_{rid}"):
                            ok, msg = delete_request(rid, sid_int)
                            if ok:
                                st.success(msg)
                                if st.session_state.get('last_request') == rid:
                                    st.session_state.pop('last_request', None)
                                safe_rerun()
                            else:
                                st.warning(msg)
                    with col2:
                        if allowed:
                            with st.expander(f"Edit request {rid}"):
                                with st.form(f"edit_form_{rid}"):
                                    existing = [c.strip() for c in str(courses).split(';') if c.strip()]
                                    default_choices = [c for c in existing if c in allowed]
                                    choices = st.multiselect("Edit courses for this request", options=allowed, default=default_choices)
                                    update = st.form_submit_button("Update request")
                                    if update:
                                        if not choices:
                                            st.warning("Choose at least one course")
                                        else:
                                            ok, msg = update_request(rid, sid_int, ';'.join(choices))
                                            if ok:
                                                st.success(msg)
                                                st.session_state['last_request'] = rid
                                                safe_rerun()
                                            else:
                                                st.warning(msg)
                        else:
                            st.info("No allowed courses available for editing. Please delete the pending request or contact your advisor.")

            else:
                if allowed:
                    with st.form('course_request_form'):
                        choices = st.multiselect("Select courses to register", options=allowed)
                        submit_req = st.form_submit_button("Submit registration request")

                        if submit_req:
                            if not choices:
                                st.warning("Choose at least one course")
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
                                    'Status': 'Pending'
                                }
                                ok = append_request(request_row)
                                if ok:
                                    st.success(f"Request submitted ({req_id}). Your advisor will be notified.")
                                    # keep user logged in and avoid re-login
                                    st.session_state['last_request'] = req_id
                                    safe_rerun()
                else:
                    st.info("No allowed courses available to select (all are locked or already completed).")

elif portal == "Advisor Portal":
    st.header("Advisor Portal")
    st.write("Login with your Advisor ID and verification code.")

    # Advisor login persistence
    if st.session_state.get('advisor_id') is None:
        with st.form('advisor_login_form'):
            aid = st.text_input("Advisor ID", max_chars=6)
            avcode = st.text_input("Verification code (student count)")
            a_login = st.form_submit_button("Login as Advisor")

        if a_login:
            try:
                aid_int = int(aid)
                avcode_int = int(avcode)
            except Exception:
                st.error("Please provide numeric Advisor ID and verification code")
                st.stop()

            adrow = advisor_sys.advisors_df[advisor_sys.advisors_df['Advisor_ID'] == aid_int]
            if adrow.empty:
                st.error("Advisor ID not found")
                st.stop()
            expected = int(adrow['Student_Count'].values[0])
            if avcode_int != expected:
                st.error("Verification failed")
                st.stop()

            # persist advisor
            st.session_state['advisor_id'] = aid_int
            st.session_state['advisor_name'] = adrow['Advisor_Name'].values[0]
            st.session_state['advisor_count'] = expected
            # set current advisor context in the backend as well
            advisor_sys.current_advisor = {'id': aid_int, 'name': st.session_state['advisor_name'], 'student_count': expected}
            st.success("Logged in as advisor. Loading...")
            safe_rerun()

    else:
        aid_int = st.session_state['advisor_id']
        adname = st.session_state.get('advisor_name', '')
        st.success(f"Welcome Dr. {adname}")
        # make sure backend context is set
        advisor_sys.current_advisor = {'id': aid_int, 'name': adname, 'student_count': st.session_state.get('advisor_count')}

        if st.button('Logout (Advisor)'):
            st.session_state.pop('advisor_id', None)
            st.session_state.pop('advisor_name', None)
            st.session_state.pop('advisor_count', None)
            advisor_sys.current_advisor = None
            safe_rerun()

        # Show options
        option = st.selectbox("Choose action", ["Overview Dashboard", "Student List", "Generate Risk Report", "Manage Requests", "Export Student List"]) 

        students_df = advisor_sys.get_advisor_students()

        if option == "Overview Dashboard":
            st.subheader("Dashboard Overview")
            if students_df.empty:
                st.info("No students assigned to you.")
            else:
                # CGPA histogram
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(students_df['CGPA'].dropna(), bins=10, color='skyblue', edgecolor='black')
                ax.set_title('CGPA Distribution')
                ax.set_xlabel('CGPA')
                ax.set_ylabel('Count')
                st.pyplot(fig)

                # Payment status
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                students_df['Payment_Status'].value_counts().plot(kind='bar', ax=ax2, color=['green', 'red'])
                ax2.set_title('Payment Status')
                st.pyplot(fig2)

                # Attendance risk
                risk_count = advisor_sys.get_attendance_risk_count(students_df)
                st.metric("Students at attendance risk (<60%)", risk_count)

        elif option == "Student List":
            st.subheader("Your Students")
            if students_df.empty:
                st.info("No students")
            else:
                st.dataframe(students_df)
                sid_view = st.text_input("Enter Student ID to view profile (optional)")
                if sid_view:
                    try:
                        sidv = int(sid_view)
                        student = students_df[students_df['Student_ID']==sidv]
                        if not student.empty:
                            st.write(student.T)
                        else:
                            st.error("Student not found in your group")
                    except Exception:
                        st.error("Invalid ID")

        elif option == "Generate Risk Report":
            st.subheader("AT-RISK STUDENTS")
            if students_df.empty:
                st.info("No students to analyze")
            else:
                risk_list = []
                for _, r in students_df.iterrows():
                    factors = []
                    if r['CGPA'] < 2.0:
                        factors.append(f"Low CGPA ({r['CGPA']:.2f})")
                    if r['Payment_Status'] == 'Unpaid':
                        factors.append("Unpaid fees")
                    if pd.notna(r.get('Locked_Courses')) and r.get('Locked_Courses') not in ['None','']:
                        factors.append(f"Locked: {r.get('Locked_Courses')}")

                    # attendance
                    for col in r.index:
                        if '_Attendance' in col and pd.notna(r[col]):
                            try:
                                av = float(str(r[col]).replace('%',''))
                                if av < 60:
                                    factors.append(f"Low attendance ({col.replace('_Attendance','')}: {av:.0f}%)")
                                    break
                            except: pass

                    if factors:
                        risk_list.append({'ID': r['Student_ID'], 'Name': r['Name'], 'CGPA': r['CGPA'], 'Risks': factors})

                if not risk_list:
                    st.success("No at-risk students identified")
                else:
                    for s_r in risk_list:
                        st.markdown(f"**{s_r['Name']} (ID: {s_r['ID']}) — CGPA: {s_r['CGPA']:.2f}**")
                        for f in s_r['Risks']:
                            st.write(f" - {f}")
                        st.write('')

        elif option == "Manage Requests":
            st.subheader("Pending registration requests")
            if os.path.exists(REQUESTS_FILE):
                df_reqs = pd.read_csv(REQUESTS_FILE)
                my_reqs = df_reqs[df_reqs['Advisor_ID']==aid_int]
                pending = my_reqs[my_reqs['Status']=='Pending']
                if pending.empty:
                    st.info("No pending requests")
                else:
                    for idx, row in pending.iterrows():
                        st.markdown(f"**Request {row['Request_ID']} — {row['Student_Name']} (ID: {row['Student_ID']})**")
                        st.write(f"Courses: {row['Courses']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Approve_{row['Request_ID']}"):
                                df_reqs.loc[df_reqs['Request_ID']==row['Request_ID'],'Status'] = 'Approved'
                                df_reqs.to_csv(REQUESTS_FILE, index=False)
                                st.success('Approved')
                                safe_rerun()
                        with col2:
                            if st.button(f"Reject_{row['Request_ID']}"):
                                df_reqs.loc[df_reqs['Request_ID']==row['Request_ID'],'Status'] = 'Rejected'
                                df_reqs.to_csv(REQUESTS_FILE, index=False)
                                st.warning('Rejected')
                                safe_rerun()

                # Archive/Clear requests admin control (advisor only)
                with st.expander("Archive and clear all requests (Advisor only)"):
                    st.warning("This will archive the current requests file and reset it. This action is irreversible.")
                    confirm = st.checkbox("I understand and want to archive and clear all requests", key="confirm_archive")
                    if confirm:
                        if st.button("Archive & Clear Requests", key="archive_clear"):
                            ok, msg = archive_and_clear_requests()
                            if ok:
                                st.success(msg)
                                safe_rerun()
                            else:
                                st.error(msg)
            else:
                st.info('No requests file found')

        elif option == "Export Student List":
            st.subheader("Export")
            if students_df.empty:
                st.info("No students to export")
            else:
                fname = f"advisor_{aid_int}_students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                students_df.to_csv(fname, index=False)
                st.success(f"Exported to {fname}")

# Footer note
st.markdown("---")
st.markdown("© 2026 UNIPATH University Registration System")