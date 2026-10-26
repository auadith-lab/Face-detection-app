import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("Face Detection Application")
st.write("Upload an image below to detect and highlight human faces using MediaPipe AI.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  with mp_face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.5
  ) as face_detection:
    results = face_detection.process(image_rgb)

    annotated_image = image.copy()
    face_count = 0

    if results.detections:
      face_count = len(results.detections)
      for detection in results.detections:
        mp_drawing.draw_detection(annotated_image, detection)

  annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  st.image(
      annotated_image_rgb,
      caption=f"Detected {face_count} face(s)",
      use_container_width=True,
  )
