import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.base_layout import (
    style_background_dashboard,
    style_base_layout,
)

from src.components.headers import header_dashboard
from src.components.footer import footer_dashboard
from src.components.subject_card import subject_card

from src.components.dialog_create_subject import (
    create_subject_dialog,
)

from src.components.dialog_share_subject import (
    share_subject_dialog,
)

from src.components.dialog_add_photo import (
    add_photos_dialog,
)

from src.components.dialog_attendance_results import (
    attendance_result_dialog,
)

from src.components.dialog_voice_attendance import (
    voice_attendance_dialog,
)

from src.pipelines.face_pipeline import (
    predict_attendance,
)

from src.database.db import (
    check_teacher_exists,
    create_teacher,
    teacher_login,
    reset_teacher_password,
    get_teacher_subjects,
    get_attendance_for_teacher,
)

from src.database.config import supabase


# =========================================================
# TEACHER SCREEN
# =========================================================

def teacher_screen():

    style_background_dashboard()
    style_base_layout()

    if "teacher_data" in st.session_state:

        teacher_dashboard()

    elif (
        "teacher_login_type" not in st.session_state
        or st.session_state.teacher_login_type == "login"
    ):

        teacher_screen_login()

    elif st.session_state.teacher_login_type == "register":

        teacher_screen_register()


# =========================================================
# TEACHER DASHBOARD
# =========================================================

def teacher_dashboard():

    teacher_data = st.session_state.teacher_data

    col1, col2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge",
    )

    with col1:
        header_dashboard()

    with col2:

        st.subheader(
            f"Welcome, {teacher_data['name']}"
        )

        if st.button(
            "Logout",
            type="secondary",
            key="teacher_logout",
        ):

            st.session_state.is_logged_in = False
            st.session_state.user_role = None

            st.session_state.pop(
                "teacher_data",
                None,
            )

            st.session_state.pop(
                "current_teacher_tab",
                None,
            )

            st.session_state.teacher_login_type = "login"

            st.rerun()

    st.space()

    if "current_teacher_tab" not in st.session_state:

        st.session_state.current_teacher_tab = (
            "take_attendance"
        )

    tab1, tab2, tab3 = st.columns(3)

    # -----------------------------------------------------
    # TAKE ATTENDANCE TAB
    # -----------------------------------------------------

    with tab1:

        button_type = (
            "primary"
            if st.session_state.current_teacher_tab
            == "take_attendance"
            else "tertiary"
        )

        if st.button(
            "Take Attendance",
            type=button_type,
            width="stretch",
            icon=":material/ar_on_you:",
            key="take_attendance_tab",
        ):

            st.session_state.current_teacher_tab = (
                "take_attendance"
            )

            st.rerun()

    # -----------------------------------------------------
    # MANAGE SUBJECTS TAB
    # -----------------------------------------------------

    with tab2:

        button_type = (
            "primary"
            if st.session_state.current_teacher_tab
            == "manage_subjects"
            else "tertiary"
        )

        if st.button(
            "Manage Subjects",
            type=button_type,
            width="stretch",
            icon=":material/book_ribbon:",
            key="manage_subjects_tab",
        ):

            st.session_state.current_teacher_tab = (
                "manage_subjects"
            )

            st.rerun()

    # -----------------------------------------------------
    # ATTENDANCE RECORDS TAB
    # -----------------------------------------------------

    with tab3:

        button_type = (
            "primary"
            if st.session_state.current_teacher_tab
            == "attendance_records"
            else "tertiary"
        )

        if st.button(
            "Attendance Records",
            type=button_type,
            width="stretch",
            icon=":material/cards_stack:",
            key="attendance_records_tab",
        ):

            st.session_state.current_teacher_tab = (
                "attendance_records"
            )

            st.rerun()

    st.divider()

    if (
        st.session_state.current_teacher_tab
        == "take_attendance"
    ):

        teacher_tab_take_attendance()

    elif (
        st.session_state.current_teacher_tab
        == "manage_subjects"
    ):

        teacher_tab_manage_subjects()

    elif (
        st.session_state.current_teacher_tab
        == "attendance_records"
    ):

        teacher_tab_attendance_records()

    footer_dashboard()


# =========================================================
# TAKE ATTENDANCE
# =========================================================

def teacher_tab_take_attendance():

    teacher_id = st.session_state.teacher_data[
        "teacher_id"
    ]

    st.header("Take AI Attendance")

    if "attendance_images" not in st.session_state:

        st.session_state.attendance_images = []

    subjects = get_teacher_subjects(
        teacher_id
    )

    if not subjects:

        st.warning(
            "You haven't created any subjects yet! "
            "Please create one to begin."
        )

        return

    subject_options = {
        f"{subject['name']} - "
        f"{subject['subject_code']}":
        subject["subject_id"]
        for subject in subjects
    }

    col1, col2 = st.columns(
        [3, 1],
        vertical_alignment="bottom",
    )

    with col1:

        selected_subject_label = st.selectbox(
            "Select Subject",
            options=list(
                subject_options.keys()
            ),
        )

    with col2:

        if st.button(
            "Add Photos",
            type="primary",
            icon=":material/photo_prints:",
            width="stretch",
            key="add_attendance_photos",
        ):

            add_photos_dialog()

    selected_subject_id = subject_options[
        selected_subject_label
    ]

    st.divider()

    images = st.session_state.attendance_images

    if images:

        st.header("Added Photos")

        gallery_cols = st.columns(4)

        for idx, img in enumerate(images):

            with gallery_cols[idx % 4]:

                st.image(
                    img,
                    width="stretch",
                    caption=f"Photo {idx + 1}",
                )

    has_photos = bool(images)

    c1, c2, c3 = st.columns(3)

    # -----------------------------------------------------
    # CLEAR PHOTOS
    # -----------------------------------------------------

    with c1:

        if st.button(
            "Clear all photos",
            width="stretch",
            type="tertiary",
            icon=":material/delete:",
            disabled=not has_photos,
            key="clear_attendance_photos",
        ):

            st.session_state.attendance_images = []

            st.rerun()

    # -----------------------------------------------------
    # FACE ATTENDANCE
    # -----------------------------------------------------

    with c2:

        if st.button(
            "Run Face Analysis",
            width="stretch",
            type="secondary",
            icon=":material/analytics:",
            disabled=not has_photos,
            key="run_face_analysis",
        ):

            with st.spinner(
                "Deep scanning classroom photos..."
            ):

                all_detected_ids = {}

                # -----------------------------------------
                # PROCESS PHOTOS
                # -----------------------------------------

                for idx, img in enumerate(images):

                    img_np = np.asarray(
                        img.convert("RGB")
                    )

                    detected, _, _ = (
                        predict_attendance(
                            img_np
                        )
                    )

                    if detected:

                        for sid in detected.keys():

                            student_id = int(sid)

                            all_detected_ids.setdefault(
                                student_id,
                                [],
                            ).append(
                                f"Photo {idx + 1}"
                            )

                # -----------------------------------------
                # GET ENROLLED STUDENTS
                # -----------------------------------------

                enrolled_res = (
                    supabase
                    .table("subject_students")
                    .select("*, students(*)")
                    .eq(
                        "subject_id",
                        selected_subject_id,
                    )
                    .execute()
                )

                enrolled_students = (
                    enrolled_res.data or []
                )

                if not enrolled_students:

                    st.warning(
                        "No students enrolled in this course."
                    )

                else:

                    results = []
                    attendance_to_log = []

                    current_timestamp = (
                        datetime.now()
                        .replace(microsecond=0)
                        .isoformat()
                    )

                    # -------------------------------------
                    # BUILD RESULTS
                    # -------------------------------------

                    for node in enrolled_students:

                        student = node.get(
                            "students"
                        )

                        if not student:
                            continue

                        student_id = int(
                            student["student_id"]
                        )

                        sources = (
                            all_detected_ids.get(
                                student_id,
                                [],
                            )
                        )

                        is_present = bool(
                            sources
                        )

                        results.append(
                            {
                                "Name": student["name"],
                                "ID": student_id,
                                "Source": (
                                    ", ".join(sources)
                                    if is_present
                                    else "-"
                                ),
                                "Status": (
                                    "✅ Present"
                                    if is_present
                                    else "❌ Absent"
                                ),
                            }
                        )

                        attendance_to_log.append(
                            {
                                "student_id": student_id,
                                "subject_id": selected_subject_id,
                                "timestamp": current_timestamp,
                                "is_present": is_present,
                            }
                        )

                    # -------------------------------------
                    # SAVE RESULT IN SESSION
                    # -------------------------------------

                    st.session_state.face_attendance_results = (
                        pd.DataFrame(results),
                        attendance_to_log,
                    )

                    st.rerun()

    # -----------------------------------------------------
    # VOICE ATTENDANCE
    # -----------------------------------------------------

    with c3:

        if st.button(
            "Use Voice Attendance",
            type="primary",
            width="stretch",
            icon=":material/mic:",
            key="voice_attendance_button",
        ):

            voice_attendance_dialog(
                selected_subject_id
            )

    # -----------------------------------------------------
    # SHOW FACE ATTENDANCE RESULT
    # -----------------------------------------------------

    if st.session_state.get(
        "face_attendance_results"
    ):

        st.divider()

        df_results, logs = (
            st.session_state.face_attendance_results
        )

        attendance_result_dialog(
            df_results,
            logs,
        )


# =========================================================
# MANAGE SUBJECTS
# =========================================================

def teacher_tab_manage_subjects():

    teacher_id = st.session_state.teacher_data[
        "teacher_id"
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.header("Manage Subjects")

    with col2:

        if st.button(
            "Create New Subject",
            width="stretch",
            key="create_subject_button",
        ):

            create_subject_dialog(
                teacher_id
            )

    # -----------------------------------------------------
    # GET SUBJECTS
    # -----------------------------------------------------

    subjects = get_teacher_subjects(
        teacher_id
    )

    if not subjects:

        st.info(
            "NO SUBJECTS FOUND. "
            "CREATE ONE ABOVE."
        )

        return

    # -----------------------------------------------------
    # DISPLAY EVERY SUBJECT
    # -----------------------------------------------------

    for sub in subjects:

        stats = [
            (
                "🫂",
                "Students",
                sub.get(
                    "total_students",
                    0,
                ),
            ),
            (
                "🕰️",
                "Classes",
                sub.get(
                    "total_classes",
                    0,
                ),
            ),
        ]

        # ---------------------------------------------
        # SHARE BUTTON
        # ---------------------------------------------

        def share_btn(
            subject_name=sub["name"],
            subject_code=sub["subject_code"],
        ):

            if st.button(
                f"Share Code: {subject_name}",
                key=f"share_{subject_code}",
                icon=":material/share:",
                width="stretch",
            ):

                share_subject_dialog(
                    subject_name,
                    subject_code,
                )

        # ---------------------------------------------
        # SUBJECT CARD
        # ---------------------------------------------

        subject_card(
            name=sub["name"],
            code=sub["subject_code"],
            section=sub["section"],
            stats=stats,
            footer_callback=share_btn,
        )

        st.space()


# =========================================================
# ATTENDANCE RECORDS
# =========================================================

def teacher_tab_attendance_records():

    st.header("Attendance Records")

    teacher_id = st.session_state.teacher_data[
        "teacher_id"
    ]

    records = get_attendance_for_teacher(
        teacher_id
    )

    if not records:

        st.info(
            "No attendance records found."
        )

        return

    data = []

    for record in records:

        timestamp = record.get(
            "timestamp"
        )

        subject = (
            record.get("subjects")
            or {}
        )

        # ---------------------------------------------
        # FORMAT TIME
        # ---------------------------------------------

        if timestamp:

            try:

                dt = datetime.fromisoformat(
                    timestamp.replace(
                        "Z",
                        "+00:00",
                    )
                )

                display_time = dt.strftime(
                    "%Y-%m-%d %I:%M %p"
                )

            except Exception:

                display_time = timestamp

        else:

            display_time = "N/A"

        # ---------------------------------------------
        # GROUP SESSION
        # ---------------------------------------------

        if timestamp:

            try:

                session_dt = datetime.fromisoformat(
                    timestamp.replace(
                        "Z",
                        "+00:00",
                    )
                )

                ts_group = session_dt.replace(
                    microsecond=0
                ).isoformat()

            except Exception:

                ts_group = timestamp

        else:

            ts_group = None

        data.append(
            {
                "ts_group": ts_group,
                "Time": display_time,
                "Subject": subject.get(
                    "name",
                    "Unknown",
                ),
                "Subject Code": subject.get(
                    "subject_code",
                    "-",
                ),
                "is_present": bool(
                    record.get(
                        "is_present",
                        False,
                    )
                ),
            }
        )

    if not data:

        st.info(
            "No attendance records found."
        )

        return

    df = pd.DataFrame(data)

    # ---------------------------------------------
    # GROUP ATTENDANCE SESSIONS
    # ---------------------------------------------

    summary = (
        df
        .groupby(
            [
                "ts_group",
                "Time",
                "Subject",
                "Subject Code",
            ]
        )
        .agg(
            Present_Count=(
                "is_present",
                "sum",
            ),
            Total_Count=(
                "is_present",
                "count",
            ),
        )
        .reset_index()
    )

    summary["Attendance Stats"] = (
        "✅ "
        + summary["Present_Count"].astype(str)
        + " / "
        + summary["Total_Count"].astype(str)
        + " Students"
    )

    display_df = (
        summary
        .sort_values(
            by="ts_group",
            ascending=False,
        )
        [
            [
                "Time",
                "Subject",
                "Subject Code",
                "Attendance Stats",
            ]
        ]
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )


# =========================================================
# TEACHER LOGIN
# =========================================================

def login_teacher(
    username,
    password,
):

    if not username or not password:

        return False

    teacher = teacher_login(
        username,
        password,
    )

    if teacher:

        st.session_state.user_role = (
            "teacher"
        )

        st.session_state.teacher_data = (
            teacher
        )

        st.session_state.is_logged_in = True

        return True

    return False


# =========================================================
# TEACHER LOGIN SCREEN
# =========================================================

def teacher_screen_login():

    col1, col2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge",
    )

    with col1:

        header_dashboard()

    with col2:

        if st.button(
            "Go back to Home",
            type="secondary",
            key="teacher_login_back",
        ):

            st.session_state.login_type = None

            st.rerun()

    st.header(
        "Login using password",
        text_alignment="center",
    )

    st.space()
    st.space()

    # =====================================================
    # LOGIN FIELDS
    # =====================================================

    teacher_username = st.text_input(
        "Enter username",
        placeholder="ananyaroy",
    ).strip()

    teacher_pass = st.text_input(
        "Enter password",
        type="password",
        placeholder="Enter password",
    )

    st.divider()

    btnc1, btnc2 = st.columns(2)

    # -----------------------------------------------------
    # LOGIN BUTTON
    # -----------------------------------------------------

    with btnc1:

        if st.button(
            "Login",
            icon=":material/passkey:",
            width="stretch",
            key="teacher_login_button",
        ):

            if login_teacher(
                teacher_username,
                teacher_pass,
            ):

                st.toast(
                    "Welcome back! 👋"
                )

                time.sleep(1)

                st.rerun()

            else:

                st.error(
                    "Invalid username and password combo."
                )

    # -----------------------------------------------------
    # REGISTER BUTTON
    # -----------------------------------------------------

    with btnc2:

        if st.button(
            "Register Instead",
            type="primary",
            icon=":material/passkey:",
            width="stretch",
            key="teacher_register_button",
        ):

            st.session_state.teacher_login_type = (
                "register"
            )

            st.rerun()

    # =====================================================
    # FORGOT PASSWORD BUTTON
    # =====================================================

    st.space()

    if st.button(
        "Forgot Password?",
        type="tertiary",
        width="stretch",
        key="forgot_password_button",
    ):

        st.session_state.show_forgot_password = True

        st.rerun()

    # =====================================================
    # FORGOT PASSWORD FORM
    # =====================================================

    if st.session_state.get(
        "show_forgot_password",
        False,
    ):

        st.divider()

        st.subheader(
            "Reset Password"
        )

        reset_username = st.text_input(
            "Enter your username",
            key="reset_username",
        ).strip()

        new_password = st.text_input(
            "Enter new password",
            type="password",
            key="reset_new_password",
        )

        confirm_password = st.text_input(
            "Confirm new password",
            type="password",
            key="reset_confirm_password",
        )

        reset_col1, reset_col2 = st.columns(2)

        # -------------------------------------------------
        # RESET PASSWORD
        # -------------------------------------------------

        with reset_col1:

            if st.button(
                "Reset Password",
                type="primary",
                width="stretch",
                key="reset_password_button",
            ):

                if (
                    not reset_username
                    or not new_password
                    or not confirm_password
                ):

                    st.error(
                        "All fields are required."
                    )

                elif new_password != confirm_password:

                    st.error(
                        "Passwords do not match."
                    )

                elif not check_teacher_exists(
                    reset_username
                ):

                    st.error(
                        "Username not found."
                    )

                else:

                    try:

                        reset_teacher_password(
                            reset_username,
                            new_password,
                        )

                        st.success(
                            "Password reset successfully! "
                            "Please login with your new password."
                        )

                        st.session_state.show_forgot_password = (
                            False
                        )

                        # Clear reset fields

                        st.session_state.pop(
                            "reset_username",
                            None,
                        )

                        st.session_state.pop(
                            "reset_new_password",
                            None,
                        )

                        st.session_state.pop(
                            "reset_confirm_password",
                            None,
                        )

                        time.sleep(1.5)

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Password reset failed: {e}"
                        )

        # -------------------------------------------------
        # CANCEL RESET
        # -------------------------------------------------

        with reset_col2:

            if st.button(
                "Cancel",
                width="stretch",
                key="cancel_password_reset",
            ):

                st.session_state.show_forgot_password = (
                    False
                )

                st.session_state.pop(
                    "reset_username",
                    None,
                )

                st.session_state.pop(
                    "reset_new_password",
                    None,
                )

                st.session_state.pop(
                    "reset_confirm_password",
                    None,
                )

                st.rerun()

    footer_dashboard()


# =========================================================
# TEACHER REGISTRATION
# =========================================================

def register_teacher(
    teacher_username,
    teacher_name,
    teacher_pass,
    teacher_pass_confirm,
):

    if (
        not teacher_username
        or not teacher_name
        or not teacher_pass
    ):

        return (
            False,
            "All Fields are required!",
        )

    if check_teacher_exists(
        teacher_username
    ):

        return (
            False,
            "Username already taken.",
        )

    if teacher_pass != teacher_pass_confirm:

        return (
            False,
            "Password doesn't match.",
        )

    try:

        create_teacher(
            teacher_username,
            teacher_pass,
            teacher_name,
        )

        return (
            True,
            "Successfully Created! Login Now.",
        )

    except Exception as e:

        return (
            False,
            f"Unexpected Error: {e}",
        )


def teacher_screen_register():

    col1, col2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge",
    )

    with col1:

        header_dashboard()

    with col2:

        if st.button(
            "Go back to Home",
            type="secondary",
            key="teacher_register_back",
        ):

            st.session_state.login_type = None

            st.rerun()

    st.header(
        "Register your teacher profile"
    )

    st.space()
    st.space()

    teacher_username = st.text_input(
        "Enter username",
        placeholder="ananyaroy",
    ).strip()

    teacher_name = st.text_input(
        "Enter name",
        placeholder="Ananya Roy",
    ).strip()

    teacher_pass = st.text_input(
        "Enter password",
        type="password",
        placeholder="Enter password",
    )

    teacher_pass_confirm = st.text_input(
        "Confirm your password",
        type="password",
        placeholder="Enter password",
    )

    st.divider()

    btnc1, btnc2 = st.columns(2)

    # -----------------------------------------------------
    # REGISTER
    # -----------------------------------------------------

    with btnc1:

        if st.button(
            "Register now",
            icon=":material/passkey:",
            width="stretch",
            key="teacher_register_now",
        ):

            success, message = register_teacher(
                teacher_username,
                teacher_name,
                teacher_pass,
                teacher_pass_confirm,
            )

            if success:

                st.success(message)

                time.sleep(2)

                st.session_state.teacher_login_type = (
                    "login"
                )

                st.rerun()

            else:

                st.error(message)

    # -----------------------------------------------------
    # LOGIN INSTEAD
    # -----------------------------------------------------

    with btnc2:

        if st.button(
            "Login Instead",
            type="primary",
            icon=":material/passkey:",
            width="stretch",
            key="teacher_login_instead",
        ):

            st.session_state.teacher_login_type = (
                "login"
            )

            st.rerun()

    footer_dashboard()

