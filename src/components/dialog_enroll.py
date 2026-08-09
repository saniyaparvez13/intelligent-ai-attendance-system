import streamlit as st

from src.database.db import enroll_student_to_subject
from src.database.config import supabase


@st.dialog("Enroll in Subject")
def enroll_dialog():

    st.write(
        "Enter the subject code provided by your teacher "
        "to enroll."
    )

    join_code = st.text_input(
        "Subject Code",
        placeholder="Eg. CS101",
    ).strip().upper()

    if st.button(
        "Enroll now",
        type="primary",
        width="stretch",
    ):

        # ---------------------------------------------
        # VALIDATE SUBJECT CODE
        # ---------------------------------------------

        if not join_code:

            st.warning(
                "Please enter a subject code."
            )

            return

        # ---------------------------------------------
        # GET CURRENT STUDENT
        # ---------------------------------------------

        student_data = st.session_state.get(
            "student_data"
        )

        if not student_data:

            st.error(
                "Student session expired. "
                "Please login again."
            )

            return

        student_id = student_data["student_id"]

        # ---------------------------------------------
        # FIND SUBJECT
        # ---------------------------------------------

        try:

            res = (
                supabase
                .table("subjects")
                .select(
                    "subject_id, name, subject_code, section"
                )
                .eq(
                    "subject_code",
                    join_code,
                )
                .execute()
            )

        except Exception as e:

            st.error(
                f"Unable to find subject: {e}"
            )

            return

        # ---------------------------------------------
        # SUBJECT NOT FOUND
        # ---------------------------------------------

        if not res.data:

            st.error(
                f"No subject found with code "
                f"'{join_code}'."
            )

            return

        subject = res.data[0]

        subject_id = subject["subject_id"]

        # ---------------------------------------------
        # CHECK ALREADY ENROLLED
        # ---------------------------------------------

        try:

            check = (
                supabase
                .table("subject_students")
                .select(
                    "subject_id, student_id"
                )
                .eq(
                    "subject_id",
                    subject_id,
                )
                .eq(
                    "student_id",
                    student_id,
                )
                .execute()
            )

        except Exception as e:

            st.error(
                f"Unable to check enrollment: {e}"
            )

            return

        if check.data:

            st.warning(
                "You are already enrolled in this subject."
            )

            return

        # ---------------------------------------------
        # ENROLL STUDENT
        # ---------------------------------------------

        try:

            response = enroll_student_to_subject(
                student_id,
                subject_id,
            )

            if response is not None:

                st.success(
                    f"Successfully enrolled in "
                    f"{subject['name']} ({subject['subject_code']})!"
                )

                # Keep the current student session safe
                st.session_state.student_data = (
                    student_data
                )

                # Refresh dashboard
                st.rerun()

            else:

                st.error(
                    "Enrollment failed."
                )

        except Exception as e:

            st.error(
                f"Enrollment failed: {e}"
            )

            