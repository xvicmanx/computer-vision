import cv2 as cv

ESC_KEY = 27

def capture_webcam(name = 'Web Cam 1', deviceId = 0, callback = None):
  cv.namedWindow(name)
  cam = cv.VideoCapture(deviceId)

  if cam.isOpened():
    rval, frame = cam.read()
  else:
    rval = False

  while rval:
    if callback is not None:
      cv.imshow(name, callback(frame))

    rval, frame = cam.read()
    key = cv.waitKey(20)

    if key == ESC_KEY:
      break

  cv.destroyWindow(name)