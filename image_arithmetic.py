import numpy as np
import cv2

img1 = cv2.imread('pictures/watch.jpg')
img2 = cv2.imread('pictures/watch1.jpg')

#img = img1 + img2

img1_gray  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img1_gray, 220, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('gray', img1_gray)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()