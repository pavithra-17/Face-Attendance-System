import cv2
import mediapipe as mp
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Initialize Backends
mp_face_detection = mp.solutions.face_detection
KNOWN_FACES_DIR = "images"
ATTENDANCE_FILE = "attendance.csv"

class AttendanceGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Attendance Tracker Workspace")
        self.window.geometry("950x600")
        self.window.configure(bg="#2d2d2d")

        # Main Title
        title = tk.Label(window, text="AUTOMATED FACE ATTENDANCE SYSTEM", font=("Arial", 18, "bold"), fg="white", bg="#2d2d2d")
        title.pack(pady=10)

        # Main Layout split frame
        self.main_frame = tk.Frame(window, bg="#2d2d2d")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Frame: Camera Live Stream
        self.left_frame = tk.Frame(self.main_frame, bg="#1e1e1e", width=600, height=450)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.cam_label = tk.Label(self.left_frame, bg="#1e1e1e")
        self.cam_label.pack(fill="both", expand=True)

        # Right Frame: Attendance Log Table
        self.right_frame = tk.Frame(self.main_frame, bg="#1e1e1e", width=300, height=450)
        self.right_frame.pack(side="right", fill="both", padx=5, pady=5)

        table_title = tk.Label(self.right_frame, text="TODAY'S LOG", font=("Arial", 12, "bold"), fg="white", bg="#1e1e1e")
        table_title.pack(pady=5)

        # Table Structure
        self.tree = ttk.Treeview(self.right_frame, columns=("Name", "Time"), show="headings", height=18)
        self.tree.heading("Name", text="Registered Name")
        self.tree.heading("Time", text="Check-In Time")
        self.tree.column("Name", width=150, anchor="center")
        self.tree.column("Time", width=120, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Bottom Frame: Control Panels
        self.bottom_frame = tk.Frame(window, bg="#2d2d2d")
        self.bottom_frame.pack(fill="x", side="bottom", pady=15)

        self.btn_export = tk.Button(self.bottom_frame, text="EXPORT REPORT (EXCEL)", command=self.export_excel, font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", padx=15, pady=5)
        self.btn_export.pack(side="right", padx=20)

        # Camera Engine Setup
        self.cap = cv2.VideoCapture(0)
        self.counter = 0
        self.identity = "UNKNOWN"
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

        # Load Existing Table Records immediately
        self.load_existing_logs()

        # Start the synchronized Tkinter loop cycle instead of a separate thread
        self.update_frame()

        # Close Safely window intercept
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def log_attendance(self, name):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        if not os.path.exists(ATTENDANCE_FILE):
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
            df.to_csv(ATTENDANCE_FILE, index=False)
            
        df = pd.read_csv(ATTENDANCE_FILE)
        already_marked = ((df['Name'] == name) & (df['Date'] == date_str)).any()
        
        if not already_marked:
            new_row = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            
            # Immediately Append to Visual Table safely
            self.tree.insert("", "end", values=(name, time_str))

    def load_existing_logs(self):
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_csv(ATTENDANCE_FILE)
                today = datetime.now().strftime("%Y-%m-%d")
                today_entries = df[df['Date'] == today]
                for _, row in today_entries.iterrows():
                    self.tree.insert("", "end", values=(row['Name'], row['Time']))
            except Exception:
                pass

    def update_frame(self):
        """Native Tkinter loop handler running frame capture sequentially with window redraws."""
        if self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                h, w, _ = frame.shape
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(image_rgb)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        xmin = max(0, int(bboxC.xmin * w))
                        ymin = max(0, int(bboxC.ymin * h))
                        width = int(bboxC.width * w)
                        height = int(bboxC.height * h)
                        
                        face_crop = frame[ymin:ymin+height, xmin:xmin+width]
                        
                        # Process DeepFace matching every 15 frames to prevent logic stutter
                        if self.counter % 15 == 0 and face_crop.size > 0:
                            try:
                                dfs = DeepFace.find(img_path=face_crop, db_path=KNOWN_FACES_DIR, model_name="ArcFace", enforce_detection=False, silent=True)
                                if len(dfs) > 0 and not dfs[0].empty:
                                    file_base = os.path.basename(dfs[0]['identity'].iloc[0])
                                    raw_name = file_base.split('.')[0]
                                    self.identity = raw_name.split('_')[0].upper()
                                else:
                                    self.identity = "UNKNOWN"
                            except Exception:
                                self.identity = "UNKNOWN"
                        
                        color = (0, 255, 0) if self.identity != "UNKNOWN" else (0, 0, 255)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), color, 2)
                        cv2.putText(frame, self.identity, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        if self.identity != "UNKNOWN":
                            self.log_attendance(self.identity)
                            
                self.counter += 1

                # Convert to Tkinter PhotoImage layout
                cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_image)
                img_tk = ImageTk.PhotoImage(image=img)
                
                self.cam_label.img_tk = img_tk
                self.cam_label.configure(image=img_tk)
            
            # Request the next frame redraw precisely 15ms later (~60 FPS synchronization)
            self.window.after(15, self.update_frame)

    def export_excel(self):
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_csv(ATTENDANCE_FILE)
                excel_file = f"Attendance_Report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
                df.to_excel(excel_file, index=False)
                messagebox.showinfo("Success", f"Report saved successfully as:\n{excel_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report format: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No tracking logs found to export yet.")

    def on_close(self):
        self.cap.release()
        self.face_detection.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceGUI(root)
    root.mainloop()