import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Face Detection & Recognition App", page_icon="👤", layout="wide"
)

st.title("👤 Face Detection & Recognition App")
st.write(
    "Upload an image below to detect and highlight human faces using MediaPipe AI."
)

mp_face_detection = mp.solutions.face_detection

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, 1)
  H, W, _ = img.shape

  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  with mp_face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.5
  ) as face_detection:
    results = face_detection.process(img_rgb)

    face_count = 0
    if results.detections:
      face_count = len(results.detections)
      for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * W)
        y = int(bbox.ymin * H)
        w = int(bbox.width * W)
        h = int(bbox.height * H)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

  img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Processed Image Output")
    st.image(img_final, channels="RGB", use_container_width=True)

  with col2:
    st.subheader("Detection Analysis & Metrics")
    st.success(
        f"Successfully detected **{face_count}** face(s) using MediaPipe!"
    )
    st.markdown("""
        * **Framework:** Google MediaPipe ML
        * **Status:** Fully Operational & Online
        """)
else:
  st.info("Please upload an image file to start face detection.")
