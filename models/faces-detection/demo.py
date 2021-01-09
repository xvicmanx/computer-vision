from faces_detector import FacesDetector
from helpers import capture_webcam

detector = FacesDetector()

capture_webcam(
  'Camera 1',
  0,
  detector.detect_and_display
)
