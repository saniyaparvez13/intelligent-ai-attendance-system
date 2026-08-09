import streamlit as st
from PIL import Image


@st.dialog("Capture or upload photos")
def add_photos_dialog():

    st.write(
        "Add classroom photos to scan for attendance."
    )

    if "photo_tab" not in st.session_state:
        st.session_state.photo_tab = "camera"

    if "attendance_images" not in st.session_state:
        st.session_state.attendance_images = []

    col1, col2 = st.columns(2)

    with col1:

        camera_type = (
            "primary"
            if st.session_state.photo_tab == "camera"
            else "secondary"
        )

        if st.button(
            "Camera",
            type=camera_type,
            width="stretch"
        ):
            st.session_state.photo_tab = "camera"
            st.rerun()

    with col2:

        upload_type = (
            "primary"
            if st.session_state.photo_tab == "upload"
            else "secondary"
        )

        if st.button(
            "Upload photos",
            type=upload_type,
            width="stretch"
        ):
            st.session_state.photo_tab = "upload"
            st.rerun()

    if st.session_state.photo_tab == "camera":

        photo = st.camera_input(
            "Take Snapshot",
            key="attendance_camera"
        )

        if photo:

            image = Image.open(photo).convert("RGB")

            st.session_state.attendance_images.append(
                image
            )

            st.toast("Photo captured!")

            st.rerun()

    else:

        files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="attendance_upload"
        )

        if files:

            existing_count = len(
                st.session_state.attendance_images
            )

            for file in files:

                image = Image.open(file).convert("RGB")

                # Avoid duplicate additions during reruns
                if existing_count == 0:
                    st.session_state.attendance_images.append(
                        image
                    )

            st.toast("Photos uploaded!")

    st.divider()

    st.write(
        f"Photos added: "
        f"{len(st.session_state.attendance_images)}"
    )

    if st.button(
        "Done",
        type="primary",
        width="stretch"
    ):
        st.rerun()
        