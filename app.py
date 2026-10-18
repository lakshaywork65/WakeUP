import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from utils import calculate_ear, calculate_mar

st.set_page_config(page_title="Edge Driver Fatigue Monitor", layout="wide")

st.title("Edge-Optimized Driver Fatigue & Distraction Monitor")
st.markdown("Real-time temporal monitoring using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).")

# Sidebar configurations
st.sidebar.header("Alert Thresholds")
ear_threshold = st.sidebar.slider("EAR Threshold (Drowsiness)", 0.15, 0.30, 0.21, 0.01)
mar_threshold = st.sidebar.slider("MAR Threshold (Yawning)", 0.40, 0.80, 0.60, 0.05)
closure_duration_limit = st.sidebar.slider("Consecutive Closure Limit (s)", 1.0, 3.0, 1.5, 0.1)

run_monitor = st.checkbox("Start Live Monitoring", value=False)

col1, col2 = st.columns([2, 1])

with col1:
    video_frame_placeholder = st.empty()

with col2:
    st.subheader("Live Telemetry Log")
    telemetry_placeholder = st.empty()

# MediaPipe Face Mesh Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# State tracking
eye_closed_start_time = None
drowsy_alert_active = False

if run_monitor:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened() and run_monitor:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        status_text = "Alert & Focused"
        status_color = (0, 255, 0)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Extract left and right eye landmark sets
            left_eye_pts = [33, 160, 158, 133, 153, 144]
            right_eye_pts = [362, 385, 387, 263, 373, 380]
            mouth_pts = [78, 308, 13, 14, 87, 317]
            
            left_ear = calculate_ear(landmarks, left_eye_pts, w, h)
            right_ear = calculate_ear(landmarks, right_eye_pts, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            mar = calculate_mar(landmarks, mouth_pts, w, h)
            
            # Temporal sliding window validation for eye closure
            if avg_ear < ear_threshold:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                else:
                    elapsed_closure = time.time() - eye_closed_start_time
                    if elapsed_closure >= closure_duration_limit:
                        status_text = "DROWSINESS ALERT: Eyes Closed!"
                        status_color = (0, 0, 255)
            else:
                eye_closed_start_time = None
                
            # Yawn check
            if mar > mar_threshold:
                status_text = "WARNING: Yawn Detected!"
                status_color = (0, 165, 255)
                
            # Overlay status on video frame
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
        # Convert frame back for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        with telemetry_placeholder.container():
            st.metric("Current Status", status_text)
            st.metric("EAR Value", f"{avg_ear:.2f}" if 'avg_ear' in locals() else "N/A")
            st.metric("MAR Value", f"{mar:.2f}" if 'mar' in locals() else "N/A")
            
    cap.release()
else:
    video_frame_placeholder.info("Check 'Start Live Monitoring' in the sidebar to initialize webcam inference.")
