import numpy as np
import cv2

img = cv2.imread('pictures/watch1.jpg', cv2.IMREAD_COLOR)

px = img[55,55]

img[100:150, 100:150] = (0,0,0)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



