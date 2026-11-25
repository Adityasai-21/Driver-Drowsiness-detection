import streamlit as st
import cv2
import numpy as np
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered")
st.title("🚗 Driver Drowsiness Detection")
st.write("Live eye-closure monitoring using MediaPipe Face Mesh and Eye Aspect Ratio (EAR).")
st.warning("Demo only—not a safety-certified driver monitoring system. Do not rely on it while driving.")

EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 25
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def ear(points):
    p = np.asarray(points, dtype=np.float32)
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal else 0.0

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_count = 0
        self.alert = False
        self.ear_value = 0.0
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        self.alert = False
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            h, w = img.shape[:2]
            l = [(lm[i].x*w, lm[i].y*h) for i in LEFT_EYE]
            r = [(lm[i].x*w, lm[i].y*h) for i in RIGHT_EYE]
            self.ear_value = (ear(l)+ear(r))/2
            for eye in (l,r):
                contour = np.array([(int(x),int(y)) for x,y in eye], dtype=np.int32)
                cv2.polylines(img,[contour],True,(0,255,0),2)
            if self.ear_value < EAR_THRESHOLD:
                self.closed_count += 1
            else:
                self.closed_count = 0
            self.alert = self.closed_count >= CLOSED_FRAMES
            status = "DROWSINESS ALERT!" if self.alert else "Eyes open / monitoring"
            color = (0,0,255) if self.alert else (0,255,0)
            cv2.putText(img,status,(20,45),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            cv2.putText(img,f"EAR: {self.ear_value:.2f}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        else:
            self.closed_count = 0
            cv2.putText(img,"No face detected",(20,45),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.sidebar.header("Settings")
EAR_THRESHOLD = st.sidebar.slider("Eye closure threshold (EAR)",0.10,0.40,0.25,0.01)
CLOSED_FRAMES = st.sidebar.slider("Frames before alert",5,60,25,5)
st.caption("Keep your face centered and well-lit. Start the camera below; allow browser camera access.")
ctx = webrtc_streamer(
    key="drowsiness-monitor",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
    async_processing=True,
)
st.info("When the eyes remain closed for the configured duration, the video overlay displays an alert. This browser-based camera stream is processed in the app session.")
