import streamlit as st

from src.database.db import create_attendance


# =========================================================
# ATTENDANCE RESULT
# =========================================================

def show_attendance_result(df, logs):

    st.divider()

    st.subheader("Attendance Reports")

    st.write(
        "Please review attendance before confirming."
    )

    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
    )

    col1, col2 = st.columns(2)

    # =====================================================
    # DISCARD
    # =====================================================

    with col1:

        if st.button(
            "Discard",
            width="stretch",
            key="discard_attendance",
        ):

            st.session_state.face_attendance_results = None
            st.session_state.voice_attendance_results = None
            st.session_state.attendance_images = []

            st.rerun()

    # =====================================================
    # CONFIRM & SAVE
    # =====================================================

    with col2:

        if st.button(
            "Confirm & Save",
            width="stretch",
            type="primary",
            key="confirm_save_attendance",
        ):

            try:

                if not logs:

                    st.warning(
                        "No attendance data to save."
                    )

                else:

                    # Save attendance
                    create_attendance(logs)

                    # Clear face attendance result
                    st.session_state.face_attendance_results = None

                    # Clear voice attendance result
                    st.session_state.voice_attendance_results = None

                    # Clear uploaded photos
                    st.session_state.attendance_images = []

                    # Success message
                    st.success(
                        "✅ Attendance saved successfully!"
                    )

                    st.toast(
                        "Attendance saved successfully! 🎉"
                    )

                    import time
                    time.sleep(1.5)

                    st.rerun()

            except Exception as e:

                st.error(
                    f"Failed to save attendance: {e}"
                )


# =========================================================
# ATTENDANCE RESULT
# =========================================================
#
# IMPORTANT:
# This is intentionally NOT decorated with @st.dialog.
# This prevents the "Only one dialog is allowed" error.
# =========================================================

def attendance_result_dialog(df, logs):

    show_attendance_result(
        df,
        logs,
    )

