import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


# =========================================================
# LOAD DLIB MODELS
# =========================================================

@st.cache_resource
def load_dlib_models():

    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec 


# =========================================================
# FACE EMBEDDINGS
# =========================================================

def get_face_embeddings(image_np):

    detector, sp, facerec = load_dlib_models()

    faces = detector(image_np, 1)

    encodings = []

    for face in faces:

        shape = sp(image_np, face)

        face_descriptor = facerec.compute_face_descriptor(
            image_np,
            shape,
            1
        )

        # 128-dimensional face embedding
        encodings.append(
            np.array(face_descriptor)
        )

    return encodings


# =========================================================
# TRAIN CLASSIFIER
# =========================================================

@st.cache_resource
def get_trained_model():

    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:

        embedding = student.get(
            "face_embedding"
        )

        if embedding:

            X.append(
                np.array(
                    embedding,
                    dtype=float
                )
            )

            y.append(
                student.get("student_id")
            )

    if len(X) == 0:
        return None

    # -----------------------------------------------------
    # SVM MODEL
    # -----------------------------------------------------

    clf = None

    # SVM requires at least 2 different classes
    if len(set(y)) >= 2:

        clf = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced"
        )

        try:

            clf.fit(X, y)

        except ValueError:

            clf = None

    return {
        "clf": clf,
        "X": X,
        "y": y
    }


# =========================================================
# TRAIN / REFRESH CLASSIFIER
# =========================================================

def train_classifier():

    st.cache_resource.clear()

    model_data = get_trained_model()

    return bool(model_data)


# =========================================================
# FACE ATTENDANCE PREDICTION
# =========================================================

def predict_attendance(class_image_np):

    encodings = get_face_embeddings(
        class_image_np
    )

    detected_student = {}

    model_data = get_trained_model()

    if not model_data:

        return (
            detected_student,
            [],
            len(encodings)
        )

    clf = model_data["clf"]
    X_train = model_data["X"]
    y_train = model_data["y"]

    all_students = sorted(
        list(set(y_train))
    )

    # =====================================================
    # RECOGNITION THRESHOLD
    # =====================================================
    #
    # 0.6 is the traditional face-recognition
    # distance threshold.
    #
    # We use a stricter threshold to reduce
    # false positives.
    #
    # =====================================================

    resemblance_threshold = 0.45

    # =====================================================
    # PROCESS EVERY DETECTED FACE
    # =====================================================

    for encoding in encodings:

        predicted_id = None

        # -------------------------------------------------
        # CASE 1: MULTIPLE STUDENTS
        # -------------------------------------------------

        if clf is not None:

            predicted_id = int(
                clf.predict(
                    [encoding]
                )[0]
            )

        # -------------------------------------------------
        # CASE 2: ONLY ONE STUDENT
        # -------------------------------------------------
        #
        # IMPORTANT:
        # Do NOT automatically trust the only student.
        #
        # We still compare the actual face embedding.
        #
        # -------------------------------------------------

        else:

            distances = [
                np.linalg.norm(
                    np.array(train_embedding)
                    - encoding
                )
                for train_embedding in X_train
            ]

            if not distances:
                continue

            best_index = int(
                np.argmin(distances)
            )

            predicted_id = int(
                y_train[best_index]
            )

        # -------------------------------------------------
        # FIND EMBEDDING FOR PREDICTED STUDENT
        # -------------------------------------------------

        matching_indices = [
            i
            for i, student_id in enumerate(y_train)
            if int(student_id) == predicted_id
        ]

        if not matching_indices:
            continue

        # -------------------------------------------------
        # BEST DISTANCE FOR THIS STUDENT
        # -------------------------------------------------

        best_match_score = min(
            np.linalg.norm(
                np.array(
                    X_train[i]
                ) - encoding
            )
            for i in matching_indices
        )

        # -------------------------------------------------
        # ACCEPT ONLY IF FACE IS CLOSE ENOUGH
        # -------------------------------------------------

        if best_match_score <= resemblance_threshold:

            detected_student[
                predicted_id
            ] = True

    return (
        detected_student,
        all_students,
        len(encodings)
    )

