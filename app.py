import os
import urllib.request
import cv2
import numpy as np
import streamlit as st

# Set wide layout for a professional dashboard look
st.set_page_config(
    page_title="Professional Face Detection Dashboard",
    page_icon="👤",
    layout="wide",
)

st.title("👤 Professional Face Detection & Analytics Dashboard")
st.markdown(
    "Upload an image to perform high-precision computer vision face detection"
    " with adjustable parameters, privacy blurring, metrics breakdown, and"
    " data export capabilities."
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

# Sidebar Configuration
st.sidebar.header("⚙️ Detection Settings")
scale_factor = st.sidebar.slider(
    "Scale Factor",
    min_value=1.01,
    max_value=1.5,
    value=1.1,
    step=0.01,
    help="Higher values scan faster; lower values are more thorough.",
)
min_neighbors = st.sidebar.slider(
    "Min Neighbors (Confidence)",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Higher values increase strictness and reduce false detections.",
)

st.sidebar.markdown("---")
st.sidebar.header("🔒 Privacy & Effects")
blur_faces = st.sidebar.checkbox(
    "Blur Detected Faces",
    value=False,
    help="Applies a secure Gaussian blur to detected faces for privacy masking.",
)

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  h_orig, w_orig, _ = image.shape

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30)
  )

  # Process image based on user settings
  processed_image = image.copy()
  for i, (x, y, w, h) in enumerate(faces):
    if blur_faces:
      roi = processed_image[y : y + h, x : x + w]
      roi = cv2.GaussianBlur(roi, (99, 99), 30)
      processed_image[y : y + h, x : x + w] = roi
    else:
      cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(
          processed_image,
          f"Face {i+1}",
          (x, y - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 0),
          2,
      )

  # KPI Summary Metrics Cards
  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label="Faces Detected", value=len(faces))
  with col2:
    st.metric(label="Image Resolution", value=f"{w_orig} x {h_orig} px")
  with col3:
    status_text = (
        "Privacy Blur Active" if blur_faces else "Bounding Boxes Active"
    )
    st.metric(label="Operation Mode", value=status_text)

  st.markdown("---")

  # Display Processed Image
  processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
  st.image(
      processed_rgb,
      caption=f"Processed Output — {len(faces)} face(s) identified",
      use_container_width=True,
  )

  # Download Button for Processed Image
  success_encoded, buffer = cv2.imencode(".jpg", processed_image)
  if success_encoded:
    st.download_button(
        label="📥 Download Processed Image",
        data=buffer.tobytes(),
        file_name="processed_face_detection.jpg",
        mime="image/jpeg",
    )

  # Detailed Face Metrics Table & CSV Export
  if len(faces) > 0:
    st.subheader("📊 Detailed Face Coordinates & Metrics")
    metrics_data = []
    csv_rows = ["Face_ID,X_Coordinate,Y_Coordinate,Width,Height"]
    for i, (x, y, w, h) in enumerate(faces):
      metrics_data.append({
          "Face #": i + 1,
          "X Coordinate": x,
          "Y Coordinate": y,
          "Width (px)": w,
          "Height (px)": h,
      })
      csv_rows.append(f"{i+1},{x},{y},{w},{h}")

    st.table(metrics_data)

    # CSV Download Button
    csv_data = "\n".join(csv_rows)
    st.download_button(
        label="📥 Export Metrics Report as CSV",
        data=csv_data,
        file_name="face_metrics_report.csv",
        mime="text/csv",
    )
  else:
    st.info(
        "No faces detected with current settings. Try lowering the Min Neighbors"
        " slider in the sidebar!"
    )
