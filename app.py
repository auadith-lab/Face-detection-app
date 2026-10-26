import os
import urllib.request
import cv2
import numpy as np
import streamlit as st

st.title("Face Detection Application")
st.write("Upload an image to detect faces using OpenCV and Python.")


@st.cache_resource
def load_face_cascade():
  cascade_path = "haarcascade_frontalface_default.xml"
  if not os.path.exists(cascade_path):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
  return cv2.CascadeClassifier(cascade_path)


face_cascade = load_face_cascade()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
  )

  for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  st.image(
      image_rgb,
      caption=f"Detected {len(faces)} face(s)",
      use_container_width=True,
  )
