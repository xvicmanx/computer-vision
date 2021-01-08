import cv2 as cv

class FacesDetector:
  """Detect faces using Cascade classifiers
  """

  def __init__(self):
    # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
    self.__faces_cascade = cv.CascadeClassifier('./detectors/frontal_face_alt.xml')
    # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    self.__eyes_cascade = cv.CascadeClassifier('./detectors/eye.xml')
    # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml
    self.__smiles_cascade = cv.CascadeClassifier('./detectors/smile.xml')
  
  def detect(self, frame):
    """Detects faces using the Cascade classifier

    Args:
        frame (Image): An RGB image

    Returns:
        [Image]: Modified RGB with the detected faces being displayed on
    """    
    return self.__detect_faces(frame)

  def detect_and_display(self, frame):
    """Detects and display faces using the Cascade classifier

    Args:
        frame (Image): An RGB image

    Returns:
        [dict]: A dictionary with the detected regions of faces
        {
          'bounds': (x, y, w, h), // face bounds,
          'eyes': [
            (e1x, e1y, e1w, e1h),
            (e2x, e2y, e2w, e2h),
            ...
            (eNx, eNy, eNw, eNh),
          ], // Eyes bounds
          'smiles': [
            (s1x, s1y, s1w, s1h),
            (s2x, s2y, s2w, s2h),
            ...
            (sNx, sNy, sNw, sNh),
          ], // Smiles bounds
        }
    """    
    return self.__display_faces(
      self.__detect_faces(frame),
      frame
    )
  
  def __display_faces(self, items, frame):
    for item in items:
      (x, y, w, h) = item["bounds"]

      cv.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2,
      )

      roi = frame[y:y+h, x:x+w]

      for (ex, ey, ew, eh) in item['eyes']:
        cv.rectangle(
          roi,
          (ex, ey),
          (ex + ew, ey + eh),
          (0, 255, 0),
          2,
        )

      for (ex, ey, ew, eh) in item['smiles']:
        cv.rectangle(
          roi,
          (ex, ey),
          (ex + ew, ey + eh),
          (0, 0, 255),
          2,
        )
  
    return frame

  def __detect_faces(self, frame):
    gray_image = cv.equalizeHist(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    faces = self.__faces_cascade.detectMultiScale(gray_image)

    result = []
    for (x, y, w, h) in faces:
      result.append({
        'bounds': (x, y, w, h),
        'eyes': self.__eyes_cascade.detectMultiScale(gray_image[y:y+h, x:x+w]),
        'smiles': self.__smiles_cascade.detectMultiScale(gray_image[y:y+h, x:x+w], 2, 20),
      })

    return result
