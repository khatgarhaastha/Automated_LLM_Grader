from canvasapi import Canvas

# ========================
# CONFIGURATION - CHANGE THESE
# ========================
API_URL = "https://youruniversity.instructure.com"  # e.g., https://school.instructure.com
API_KEY = "your_canvas_api_key"  # Get this from Canvas under Account > Settings > New Access Token
COURSE_ID = 12345  # From the Canvas course URL: /courses/12345/

# ========================
# INITIALIZE CANVAS
# ========================
canvas = Canvas(API_URL, API_KEY)
course = canvas.get_course(COURSE_ID)

# ========================
# FETCH ENROLLMENTS (STUDENTS)
# ========================
print("Fetching enrollments...")
enrollments = course.get_enrollments()
id_to_name = {}

for e in enrollments:
    if hasattr(e, 'user') and 'name' in e.user:
        id_to_name[e.user_id] = e.user['name']

# ========================
# FETCH ASSIGNMENT COMMENTS
# ========================
print("Fetching assignment comments...\n")
for assignment in course.get_assignments():
    assignment_name = assignment.name
    submissions = assignment.get_submissions(include=['submission_comments'])

    for submission in submissions:
        user_id = submission.user_id
        comments = submission.submission_comments

        for comment in comments:
            student_name = id_to_name.get(user_id, f"User {user_id}")
            comment_text = comment.get('comment', '')
            print(f"{assignment_name} ; {student_name} ; {comment_text}")
