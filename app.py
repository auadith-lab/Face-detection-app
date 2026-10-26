import os
import urllib.request
import cv2
import numpy as np
import streamlit as st

st.title("Face Detection Application")
st.write(
    "Upload an image to detect faces using OpenCV with adjustable sensitivity"
    " and detailed metrics."
)


@st.cache_resource
def load_face_cascade():
  cascade_path = "haarcascade_frontalface_default.xml"
  if not os.path.exists(cascade_path):
    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    )
    urllib.request.urlretrieve(url, cascade_path)
  return cv2.CascadeClassifier(cascade_path)


face_cascade = load_face_cascade()

# Feature 1: Sensitivity / Confidence Sliders in the Sidebar
st.sidebar.header("Detection Settings")
scale_factor = st.sidebar.slider(
    "Scale Factor",
    min_value=1.01,
    max_value=1.5,
    value=1.1,
    step=0.01,
    help="Higher values scan faster; lower values are more thorough.",
)
min_neighbors = st.sidebar.slider(
    "Min Neighbors (Sensitivity / Confidence)",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help=(
        "Higher values increase strictness and reduce false detections."
    ),
)

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30)
  )

  # Draw bounding boxes and labels on faces
  for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"Face {i+1}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  st.image(
      image_rgb,
      caption=f"Detected {len(faces)} face(s)",
      use_container_width=True,
  )

  # Feature 2: Detailed Face Metrics Table
  if len(faces) > 0:
    st.subheader("📊 Detailed Face Metrics")
    metrics_data = []
    for i, (x, y, w, h) in enumerate(faces):
      metrics_data.append({
          "Face #": i + 1,
          "X Coordinate": x,
          "Y Coordinate": y,
          "Width (px)": w,
          "Height (px)": h,
      })
    st.table(metrics_data)
  else:
    st.info(
        "No faces detected with current settings. Try lowering the Min Neighbors"
        " slider in the sidebar!"
    )
