import cv2 as cv
import streamlit as st
from faces_detector import FacesDetector

detector = FacesDetector()

st.title('Face detection example')

run = st.checkbox('Run')

window = st.image([])
camera = cv.VideoCapture(0)

while run:
  _, frame = camera.read()
  frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
  window.image(frame)
else:
  st.write('Stopped')

# capture_webcam(
#   'Camera 1',
#   0,
#   detector.detect_and_display
# )
