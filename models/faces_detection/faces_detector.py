import cv2 as cv


DEFAULT_COLORS = {
  'faces': (255, 0, 0),
  'eyes': (0, 255, 0),
  'smiles': (0, 0, 255),
}

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
DEFAULT_FACES_CASCADE_PATH = './detectors/frontal_face_alt.xml'

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
DEFAULT_EYES_CASCADE_PATH = './detectors/eye.xml'

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml
DEFAULT_SMILES_CASCADE_PATH = './detectors/smile.xml'

class FacesDetector:
  """Detect faces using Cascade classifiers
  """

  def __init__(
    self,
    faces_cascade_file_path = DEFAULT_FACES_CASCADE_PATH,
    eyes_cascade_file_path = DEFAULT_EYES_CASCADE_PATH,
    smiles_cascade_file_path = DEFAULT_SMILES_CASCADE_PATH,
    faces_cascade_scale_factor = 1.3,
    faces_cascade_min_neighbors = 5,
    eyes_cascade_scale_factor = 1.3,
    eyes_cascade_min_neighbors = 5,
    smiles_cascade_scale_factor = 1.3,
    smiles_cascade_min_neighbors = 5,
  ): 
    self.__faces_cascade = cv.CascadeClassifier(faces_cascade_file_path)
    self.__eyes_cascade = cv.CascadeClassifier(eyes_cascade_file_path)
    self.__smiles_cascade = cv.CascadeClassifier(smiles_cascade_file_path)
    self.__faces_cascade_scale_factor = faces_cascade_scale_factor
    self.__faces_cascade_min_neighbors = faces_cascade_min_neighbors
    self.__eyes_cascade_scale_factor = eyes_cascade_scale_factor
    self.__eyes_cascade_min_neighbors = eyes_cascade_min_neighbors
    self.__smiles_cascade_scale_factor = smiles_cascade_scale_factor
    self.__smiles_cascade_min_neighbors = smiles_cascade_min_neighbors
  
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
  
  def __display_faces(
    self,
    items,
    frame,
    colors = {},
  ):
    for item in items:
      (x, y, w, h) = item['bounds']
      roi = frame[y:y+h, x:x+w]

      self.__draw_region(
        frame,
        item['bounds'],
        colors.get('faces', DEFAULT_COLORS['faces']),
      )
      
      for eye_bounds in item['eyes']:
        self.__draw_region(
          roi,
          eye_bounds,
          colors.get('eyes', DEFAULT_COLORS['eyes']),
        )

      for smile_bounds in item['smiles']:
        self.__draw_region(
          roi,
          smile_bounds,
          colors.get('smiles', DEFAULT_COLORS['smiles']),
        )
  
    return frame

  def __draw_region(
    self,
    target,
    bounds,
    color,
    rectangle_stroke_with = 2,
  ):
    (x, y, w, h) = bounds
    cv.rectangle(
      target,
      (x, y),
      (x + w, y + h),
      color,
      rectangle_stroke_with,
    )

  def __detect_faces(self, frame):
    gray_image = cv.equalizeHist(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    faces = self.__faces_cascade.detectMultiScale(
      gray_image,
      self.__faces_cascade_scale_factor,
      self.__faces_cascade_min_neighbors,
    )

    result = []
    for (x, y, w, h) in faces:
      result.append({
        'bounds': (x, y, w, h),
        'eyes': self.__eyes_cascade.detectMultiScale(
          gray_image[y:y+h, x:x+w],
          self.__eyes_cascade_scale_factor,
          self.__eyes_cascade_min_neighbors,
        ),
        'smiles': self.__smiles_cascade.detectMultiScale(
          gray_image[y:y+h, x:x+w],
          self.__smiles_cascade_scale_factor,
          self.__smiles_cascade_min_neighbors,
        ),
      })

    return result
