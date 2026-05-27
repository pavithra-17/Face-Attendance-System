import os
import base64
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from deepface import DeepFace
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
app.secret_key = "secure_attendance_portal_key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

KNOWN_FACES_DIR = "images"
ATTENDANCE_FILE = "attendance.csv"

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    role = db.Column(db.String(20), default="student")

with app.app_context():
    db.create_all()
    if not User.query.filter_by(email="teacher@school.com").first():
        teacher = User(name="Head Teacher", email="teacher@school.com", role="teacher")
        db.session.add(teacher)
        db.session.commit()

def process_base64_image(base64_string):
    encoded_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def log_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    cutoff_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    status = "PRESENT" if now <= cutoff_time else "LATE"
    
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv(ATTENDANCE_FILE, index=False)
        
    df = pd.read_csv(ATTENDANCE_FILE)
    already_marked = ((df['Name'] == name) & (df['Date'] == date_str)).any()
    
    if not already_marked:
        new_row = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str, "Status": status}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return status
    
    existing_row = df[(df['Name'] == name) & (df['Date'] == date_str)]
    return existing_row.iloc[0]['Status']

def get_dashboard_metrics(today_str):
    """Calculates fractions by checking CSV entries against the Master SQLite Student Roster."""
    total_registered_students = User.query.filter_by(role="student").count()
    
    if not os.path.exists(ATTENDANCE_FILE):
        return 0, 0, 0, total_registered_students, total_registered_students, []

    df = pd.read_csv(ATTENDANCE_FILE)
    today_df = df[df['Date'] == today_str]
    
    scanned_count = len(today_df)
    late_count = len(today_df[today_df['Status'] == 'LATE'])
    present_count = len(today_df[today_df['Status'] == 'PRESENT'])
    absent_count = total_registered_students - scanned_count
    
    return scanned_count, present_count, late_count, absent_count, total_registered_students, today_df

# ---- WEBSITE RENDERING ROUTES ----

@app.route('/')
def index(): return render_template('index.html')

@app.route('/login-page')
def login_page(): return render_template('login.html')

@app.route('/register-page')
def register_page(): return render_template('register.html')

@app.route('/teacher-dashboard')
def dashboard():
    if session.get('role') != 'teacher': return redirect(url_for('index'))
    
    today = datetime.now().strftime("%Y-%m-%d")
    scanned, present, late, absent, total_students, today_df = get_dashboard_metrics(today)
    
    # Default view shows everything scanned today
    logs = today_df.to_dict(orient="records")
    return render_template('dashboard.html', logs=logs, total=scanned, present=present, late=late, absent=absent, total_students=total_students, current_view="Total Scanned Today")

@app.route('/dashboard/view/<segment>')
def dashboard_segmented_view(segment):
    """Filters the active log table view based on the card clicked by the teacher."""
    if session.get('role') != 'teacher': return redirect(url_for('index'))
    
    today = datetime.now().strftime("%Y-%m-%d")
    scanned, present, late, absent, total_students, today_df = get_dashboard_metrics(today)
    
    logs = []
    view_title = "Records"
    
    if segment == "scanned":
        logs = today_df.to_dict(orient="records")
        view_title = "Total Scanned Today"
    elif segment == "ontime":
        logs = today_df[today_df['Status'] == 'PRESENT'].to_dict(orient="records")
        view_title = "On-Time Deliveries"
    elif segment == "absent":
        # Math Set Difference: Master Registry Students minus Scanned Students
        scanned_names = today_df['Name'].tolist() if not today_df.empty else []
        all_students = User.query.filter_by(role="student").all()
        for stud in all_students:
            if stud.name.upper() not in scanned_names:
                logs.append({"Name": stud.name.upper(), "Date": today, "Time": "--:--:--", "Status": "ABSENT"})
        view_title = "Absent Roster"
        
    return render_template('dashboard.html', logs=logs, total=scanned, present=present, late=late, absent=absent, total_students=total_students, current_view=view_title)

@app.route('/teacher-evaluation')
def late_evaluation_page():
    """Renders the fresh evaluation page specifically for late student management panels."""
    if session.get('role') != 'teacher': return redirect(url_for('index'))
    
    today = datetime.now().strftime("%Y-%m-%d")
    logs = []
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        late_df = df[(df['Date'] == today) & (df['Status'] == 'LATE')]
        logs = late_df.to_dict(orient="records")
        
    return render_template('evaluation.html', logs=logs)

# ---- WEB API INTERACTION ENGINES ----

@app.route('/api/evaluate-student', methods=['POST'])
def evaluate_student():
    """Overrides a student's tracking state and rewrites the CSV matrix file safely."""
    if session.get('role') != 'teacher': return jsonify({"success": False, "message": "Unauthorized"})
    
    data = request.json
    name = data.get('name', '').upper()
    action = data.get('action') # 'PRESENT' (On-Time override) or 'ABSENT' (Erase/Mark Absent)
    today = datetime.now().strftime("%Y-%m-%d")
    
    if not os.path.exists(ATTENDANCE_FILE):
        return jsonify({"success": False, "message": "No log ledger file exists"})
        
    df = pd.read_csv(ATTENDANCE_FILE)
    
    # Target row query match criteria mapping
    mask = (df['Name'] == name) & (df['Date'] == today)
    
    if action == 'PRESENT':
        # Update status to PRESENT inside the CSV row matrix
        df.loc[mask, 'Status'] = 'PRESENT'
    elif action == 'ABSENT':
        # Drop the row entirely from today's ledger log, effectively marking them absent
        df = df[~mask]
        
    df.to_csv(ATTENDANCE_FILE, index=False)
    return jsonify({"success": True})

@app.route('/api/teacher-login', methods=['POST'])
def teacher_login():
    email = request.json.get('email', '').strip().lower()
    user = User.query.filter_by(email=email, role="teacher").first()
    if user:
        session['user'] = user.name
        session['role'] = 'teacher'
        return jsonify({"success": True, "redirect": "/teacher-dashboard"})
    return jsonify({"success": False, "message": "Access Denied: Invalid Teacher Email"})

@app.route('/api/register', methods=['POST'])
def register_student():
    data = request.json
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    image_data = data.get('image')

    if not name or not email or not image_data:
        return jsonify({"success": False, "message": "All fields are required"})
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email ID is already registered"})

    try:
        frame = process_base64_image(image_data)
        mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb_frame)
        
        if not results.detections:
            return jsonify({"success": False, "message": "Registration Failed: No face detected. Try again."})

        file_name = f"{name.lower().replace(' ', '_')}_front.jpg"
        cv2.imwrite(os.path.join(KNOWN_FACES_DIR, file_name), frame)
        
        for file in os.listdir(KNOWN_FACES_DIR):
            if file.endswith('.pkl'): os.remove(os.path.join(KNOWN_FACES_DIR, file))

        new_student = User(name=name, email=email, role="student")
        db.session.add(new_student)
        db.session.commit()
        return jsonify({"success": True, "message": f"Biometric Registration Complete for {name}!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server processing matrix failure error: {str(e)}"})

@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    image_data = request.json.get('image')
    try:
        frame = process_base64_image(image_data)
        temp_path = "temp_scan.jpg"
        cv2.imwrite(temp_path, frame)

        dfs = DeepFace.find(img_path=temp_path, db_path=KNOWN_FACES_DIR, model_name="ArcFace", enforce_detection=False, silent=True)
        if os.path.exists(temp_path): os.remove(temp_path)

        if len(dfs) > 0 and not dfs[0].empty:
            # ArcFace distance threshold is typically around 0.68. 
            # Lowering this threshold value means the system gets MUCH stricter!
            best_match_distance = dfs[0]['distance'].iloc[0]
            
            if best_match_distance > 0.62:
                return jsonify({"success": False, "message": "Face match confidence too low. Access Denied."})
                
            matched_path = dfs[0]['identity'].iloc[0]
            file_base = os.path.basename(matched_path)
            
            # 1. Lowercase everything to make extension matching easy
            file_lower = file_base.lower()
            
            # 2. Cleanly strip off any common image extension formats
            for ext in ['.jpg', '.jpeg', '.png']:
                file_lower = file_lower.replace(ext, '')
                
            # 3. Strip off our custom backend directional tags if present
            clean_name_string = file_lower.replace('_front', '')
            
            # 4. Turn underscores into clean visual spaces
            identity = clean_name_string.replace('_', ' ').upper()
            
            status = log_attendance(identity)
            return jsonify({"success": True, "name": identity, "status": status})
        
        return jsonify({"success": False, "message": "Face structure match verification mismatch. Access Denied."})
    except Exception:
        return jsonify({"success": False, "message": "Biometric engine initialization error"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)