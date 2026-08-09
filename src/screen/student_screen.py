import streamlit as st

from src.ui.base_layout import (
    style_background_dashboard,
    style_base_layout,
)

from src.components.headers import header_dashboard
from src.components.footer import footer_dashboard

from PIL import Image
import numpy as np
import time

from src.pipelines.face_pipeline import (
    predict_attendance,
    get_face_embeddings,
    train_classifier,
)

from src.pipelines.voice_pipeline import (
    get_voice_embedding,
)

from src.database.db import (
    get_all_students,
    create_student,
    get_student_subjects,
    get_student_attendance,
    unenroll_student_to_subject,
)

from src.database.config import supabase

from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card


# =========================================================
# STUDENT DASHBOARD
# =========================================================

def student_dashboard():

    student_data = st.session_state.student_data
    student_id = student_data["student_id"]

    # =====================================================
    # HEADER
    # =====================================================

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge",
    )

    with c1:

        header_dashboard()

    with c2:

        st.subheader(
            f"Welcome, {student_data['name']}"
        )

        if st.button(
            "Logout",
            type="secondary",
            key="student_logout",
            shortcut="control+backspace",
        ):

            st.session_state["is_logged_in"] = False

            st.session_state.pop(
                "student_data",
                None,
            )

            st.rerun()

    st.space()

    # =====================================================
    # VOICE ENROLLMENT
    # =====================================================

    voice_embedding = student_data.get(
        "voice_embedding"
    )

    if not voice_embedding:

        with st.container(border=True):

            st.subheader(
                "🎙️ Voice Enrollment"
            )

            st.write(
                "Register your voice so AI can recognize "
                "you during Voice Attendance."
            )

            st.info(
                "Say a short phrase such as: "
                "\"I am present, my name is Saniya.\""
            )

            voice_audio = st.audio_input(
                "Record your voice",
                key="student_voice_enrollment",
            )

            if voice_audio:

                if st.button(
                    "Save Voice",
                    type="primary",
                    width="stretch",
                    key="save_student_voice",
                ):

                    with st.spinner(
                        "Processing your voice..."
                    ):

                        try:

                            audio_bytes = (
                                voice_audio.read()
                            )

                            new_voice_embedding = (
                                get_voice_embedding(
                                    audio_bytes
                                )
                            )

                            if new_voice_embedding is not None:

                                # ---------------------------------
                                # SAVE VOICE EMBEDDING
                                # ---------------------------------

                                response = (
                                    supabase
                                    .table("students")
                                    .update(
                                        {
                                            "voice_embedding": (
                                                new_voice_embedding
                                            )
                                        }
                                    )
                                    .eq(
                                        "student_id",
                                        student_id,
                                    )
                                    .execute()
                                )

                                if response.data:

                                    # Update current session
                                    # with latest student data
                                    updated_student = (
                                        response.data[0]
                                    )

                                    st.session_state.student_data = (
                                        updated_student
                                    )

                                    st.success(
                                        "✅ Voice enrolled successfully!"
                                    )

                                    st.toast(
                                        "Voice enrollment completed! 🎙️"
                                    )

                                    time.sleep(1)

                                    st.rerun()

                                else:

                                    st.error(
                                        "Could not save your voice."
                                    )

                            else:

                                st.error(
                                    "Could not extract voice features. "
                                    "Please record your voice again."
                                )

                        except Exception as e:

                            st.error(
                                f"Voice enrollment failed: {e}"
                            )

    else:

        st.success(
            "🎙️ Voice enrollment completed."
        )

    st.space()

    # =====================================================
    # SUBJECT HEADER
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.header(
            "Your Enrolled Subjects"
        )

    with c2:

        if st.button(
            "Enroll in Subject",
            type="primary",
            width="stretch",
            key="enroll_subject_button",
        ):

            enroll_dialog()

    st.divider()

    # =====================================================
    # LOAD SUBJECTS + ATTENDANCE
    # =====================================================

    with st.spinner(
        "Loading your enrolled subjects.."
    ):

        subjects = get_student_subjects(
            student_id
        )

        logs = get_student_attendance(
            student_id
        )

    # =====================================================
    # ATTENDANCE STATS
    # =====================================================

    stats_map = {}

    for log in logs:

        sid = log["subject_id"]

        if sid not in stats_map:

            stats_map[sid] = {
                "total": 0,
                "attended": 0,
            }

        stats_map[sid]["total"] += 1

        if log.get("is_present"):

            stats_map[sid]["attended"] += 1

    # =====================================================
    # SUBJECT CARDS
    # =====================================================

    cols = st.columns(2)

    for i, sub_node in enumerate(subjects):

        sub = sub_node["subjects"]

        sid = sub["subject_id"]

        stats = stats_map.get(
            sid,
            {
                "total": 0,
                "attended": 0,
            },
        )

        def unenroll_button(
            subject_id=sid,
            subject_name=sub["name"],
        ):

            if st.button(
                "Unenroll from this course",
                type="tertiary",
                width="stretch",
                icon=":material/delete_forever:",
                key=f"unenroll_{subject_id}",
            ):

                unenroll_student_to_subject(
                    student_id,
                    subject_id,
                )

                st.toast(
                    f"Unenrolled from {subject_name} successfully!"
                )

                st.rerun()

        with cols[i % 2]:

            subject_card(
                name=sub["name"],
                code=sub["subject_code"],
                section=sub["section"],
                stats=[
                    (
                        "📅",
                        "Total",
                        stats["total"],
                    ),
                    (
                        "✅",
                        "Attended",
                        stats["attended"],
                    ),
                ],
                footer_callback=unenroll_button,
            )

    # =====================================================
    # FOOTER
    # =====================================================

    footer_dashboard()


# =========================================================
# STUDENT SCREEN / FACE LOGIN
# =========================================================

def student_screen():

    style_background_dashboard()
    style_base_layout()

    # =====================================================
    # ALREADY LOGGED IN
    # =====================================================

    if "student_data" in st.session_state:

        student_dashboard()

        return

    # =====================================================
    # HEADER
    # =====================================================

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge",
    )

    with c1:

        header_dashboard()

    with c2:

        if st.button(
            "Go back to Home",
            type="secondary",
            key="student_login_back",
            shortcut="control+backspace",
        ):

            st.session_state["login_type"] = None

            st.rerun()

    # =====================================================
    # FACE LOGIN
    # =====================================================

    st.header(
        "Login using FaceID",
        text_alignment="center",
    )

    st.space()
    st.space()

    show_registration = False

    photo_source = st.camera_input(
        "Position your face in the center"
    )

    # =====================================================
    # FACE SCANNING
    # =====================================================

    if photo_source:

        img = np.array(
            Image.open(photo_source)
        )

        with st.spinner(
            "AI is scanning.."
        ):

            detected, all_ids, num_faces = (
                predict_attendance(img)
            )

            # ---------------------------------------------
            # NO FACE
            # ---------------------------------------------

            if num_faces == 0:

                st.warning(
                    "Face not found!"
                )

            # ---------------------------------------------
            # MULTIPLE FACES
            # ---------------------------------------------

            elif num_faces > 1:

                st.warning(
                    "Multiple faces found"
                )

            # ---------------------------------------------
            # SINGLE FACE
            # ---------------------------------------------

            else:

                if detected:

                    student_id = list(
                        detected.keys()
                    )[0]

                    all_students = (
                        get_all_students()
                    )

                    student = next(
                        (
                            s
                            for s in all_students
                            if s["student_id"]
                            == student_id
                        ),
                        None,
                    )

                    if student:

                        st.session_state.is_logged_in = True

                        st.session_state.user_role = (
                            "student"
                        )

                        st.session_state.student_data = (
                            student
                        )

                        st.toast(
                            f"Welcome Back {student['name']}"
                        )

                        time.sleep(1)

                        st.rerun()

                else:

                    st.info(
                        "Face not recognized! "
                        "You might be a new student!"
                    )

                    show_registration = True

    # =====================================================
    # NEW STUDENT REGISTRATION
    # =====================================================

    if show_registration:

        with st.container(border=True):

            st.header(
                "Register new Profile"
            )

            new_name = st.text_input(
                "Enter your name",
                placeholder="E.g. Hamza Rizvi",
            )

            # ---------------------------------------------
            # VOICE ENROLLMENT DURING REGISTRATION
            # ---------------------------------------------

            st.subheader(
                "Optional : Voice Enrollment"
            )

            st.info(
                "Enroll your voice for voice-only attendance."
            )

            audio_data = None

            try:

                audio_data = st.audio_input(
                    "Record a short phrase like "
                    "I am present, My name is Saniya.",
                    key="registration_voice",
                )

            except Exception:

                st.error(
                    "Audio Data failed!"
                )

            # ---------------------------------------------
            # CREATE ACCOUNT
            # ---------------------------------------------

            if st.button(
                "Create Account",
                type="primary",
                key="create_student_account",
            ):

                if new_name:

                    with st.spinner(
                        "Creating profile.."
                    ):

                        img = np.array(
                            Image.open(
                                photo_source
                            )
                        )

                        encodings = (
                            get_face_embeddings(
                                img
                            )
                        )

                        if encodings:

                            face_emb = (
                                encodings[0]
                                .tolist()
                            )

                            voice_emb = None

                            if audio_data:

                                voice_emb = (
                                    get_voice_embedding(
                                        audio_data.read()
                                    )
                                )

                            response_data = (
                                create_student(
                                    new_name,
                                    face_embedding=face_emb,
                                    voice_embedding=voice_emb,
                                )
                            )

                            if response_data:

                                train_classifier()

                                st.session_state.is_logged_in = True

                                st.session_state.user_role = (
                                    "student"
                                )

                                st.session_state.student_data = (
                                    response_data[0]
                                )

                                st.toast(
                                    f"Profile Created! "
                                    f"Hi {new_name}!"
                                )

                                time.sleep(1)

                                st.rerun()

                        else:

                            st.error(
                                "Couldnt capture your "
                                "facial features for registration"
                            )

                else:

                    st.warning(
                        "Please enter your name!"
                    )

    # =====================================================
    # FOOTER
    # =====================================================

    footer_dashboard() 

    

