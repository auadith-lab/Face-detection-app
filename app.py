import cv2
import numpy as np
import streamlit as st

st.title("Face Detection Application")
st.write(
    "Upload an image or use your webcam to detect faces using OpenCV and Python."
)

# Load the Haar Cascade classifier for face detection
@st.cache_resource
def load_face_cascade():
  return cv2.CascadeClassifier(
      cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
  )


face_cascade = load_face_cascade()

# Option to choose input method
option = st.selectbox(
    "Choose Input Method", ("Upload Image", "Live Webcam (Browser)")
)

if option == "Upload Image":
  uploaded_file = st.file_uploader(
      "Choose an image...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert BGR back to RGB for Streamlit display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(
        image_rgb,
        caption=f"Detected {len(faces)} face(s)",
        use_container_width=True,
    )

elif option == "Live Webcam (Browser)":
  # Using Streamlit's built-in camera input
  camera_image = st.camera_input("Take a picture")

  if camera_image is not None:
    bytes_data = camera_image.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
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