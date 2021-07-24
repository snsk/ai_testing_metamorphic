import cv2
import numpy as np

img = np.zeros((500, 500, 3), np.uint8)
img.fill(255)
points = np.array([(400,400), (100, 10), (50, 100)])
cv2.polylines(img, [points], True, (0, 0, 255))
  
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()