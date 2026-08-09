from src.database.config import supabase
import bcrypt


# =========================================================
# PASSWORD FUNCTIONS
# =========================================================

def hash_pass(pwd):
    return bcrypt.hashpw(
        pwd.encode(),
        bcrypt.gensalt()
    ).decode()


def check_pass(pwd, hashed):
    return bcrypt.checkpw(
        pwd.encode(),
        hashed.encode()
    )


# =========================================================
# TEACHER FUNCTIONS
# =========================================================

def check_teacher_exists(username):

    response = (
        supabase
        .table("teachers")
        .select("username")
        .eq("username", username)
        .execute()
    )

    return len(response.data) > 0


def create_teacher(username, password, name):

    data = {
        "username": username,
        "password": hash_pass(password),
        "name": name
    }

    response = (
        supabase
        .table("teachers")
        .insert(data)
        .execute()
    )

    return response.data


def teacher_login(username, password):

    response = (
        supabase
        .table("teachers")
        .select("*")
        .eq("username", username)
        .execute()
    )

    if not response.data:
        return None

    teacher = response.data[0]

    if check_pass(
        password,
        teacher["password"]
    ):
        return teacher

    return None


# =========================================================
# FORGOT PASSWORD
# =========================================================

def reset_teacher_password(username, new_password):

    response = (
        supabase
        .table("teachers")
        .update({
            "password": hash_pass(new_password)
        })
        .eq("username", username)
        .execute()
    )

    return response.data


# =========================================================
# STUDENT FUNCTIONS
# =========================================================

def get_all_students():

    response = (
        supabase
        .table("students")
        .select("*")
        .execute()
    )

    return response.data or []


def create_student(
    new_name,
    face_embedding=None,
    voice_embedding=None
):

    data = {
        "name": new_name,
        "face_embedding": face_embedding,
        "voice_embedding": voice_embedding
    }

    response = (
        supabase
        .table("students")
        .insert(data)
        .execute()
    )

    return response.data


# =========================================================
# UPDATE STUDENT VOICE
# =========================================================

def update_student_voice_embedding(
    student_id,
    voice_embedding
):

    response = (
        supabase
        .table("students")
        .update({
            "voice_embedding": voice_embedding
        })
        .eq(
            "student_id",
            student_id
        )
        .execute()
    )

    return response.data


# =========================================================
# SUBJECT FUNCTIONS
# =========================================================

def create_subject(
    subject_code,
    name,
    section,
    teacher_id
):

    data = {
        "subject_code": subject_code,
        "name": name,
        "section": section,
        "teacher_id": teacher_id
    }

    response = (
        supabase
        .table("subjects")
        .insert(data)
        .execute()
    )

    return response.data


def get_teacher_subjects(teacher_id):

    response = (
        supabase
        .table("subjects")
        .select(
            "*, subject_students(count), attendance_logs(timestamp)"
        )
        .eq("teacher_id", teacher_id)
        .execute()
    )

    subjects = response.data or []

    for sub in subjects:

        # ---------------------------------------------
        # TOTAL STUDENTS
        # ---------------------------------------------

        subject_students = sub.get(
            "subject_students",
            []
        )

        if subject_students:

            sub["total_students"] = (
                subject_students[0].get(
                    "count",
                    0
                )
            )

        else:

            sub["total_students"] = 0

        # ---------------------------------------------
        # TOTAL CLASSES
        # ---------------------------------------------

        attendance = sub.get(
            "attendance_logs",
            []
        )

        timestamps = {
            log.get("timestamp")
            for log in attendance
            if log.get("timestamp")
        }

        sub["total_classes"] = len(
            timestamps
        )

        # ---------------------------------------------
        # REMOVE RAW RELATION DATA
        # ---------------------------------------------

        sub.pop(
            "subject_students",
            None
        )

        sub.pop(
            "attendance_logs",
            None
        )

    return subjects


# =========================================================
# SUBJECT ENROLLMENT
# =========================================================

def enroll_student_to_subject(
    student_id,
    subject_id
):

    data = {
        "student_id": student_id,
        "subject_id": subject_id
    }

    response = (
        supabase
        .table("subject_students")
        .insert(data)
        .execute()
    )

    return response.data


def unenroll_student_to_subject(
    student_id,
    subject_id
):

    response = (
        supabase
        .table("subject_students")
        .delete()
        .eq(
            "student_id",
            student_id
        )
        .eq(
            "subject_id",
            subject_id
        )
        .execute()
    )

    return response.data


def get_student_subjects(student_id):

    response = (
        supabase
        .table("subject_students")
        .select("*, subjects(*)")
        .eq(
            "student_id",
            student_id
        )
        .execute()
    )

    return response.data or []


# =========================================================
# ATTENDANCE FUNCTIONS
# =========================================================

def get_student_attendance(student_id):

    response = (
        supabase
        .table("attendance_logs")
        .select("*, subjects(*)")
        .eq(
            "student_id",
            student_id
        )
        .execute()
    )

    return response.data or []


def create_attendance(logs):

    response = (
        supabase
        .table("attendance_logs")
        .insert(logs)
        .execute()
    )

    return response.data


def get_attendance_for_teacher(teacher_id):

    response = (
        supabase
        .table("attendance_logs")
        .select(
            "*, subjects!inner(*)"
        )
        .eq(
            "subjects.teacher_id",
            teacher_id
        )
        .execute()
    )

    return response.data or [] 

