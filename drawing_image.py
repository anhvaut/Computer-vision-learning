import numpy as np
import cv2

img = cv2.imread('pictures/watch.jpg', cv2.IMREAD_COLOR)

cv2.line(img, (0,0), (300,300), (0,0,0), 15)
cv2.rectangle(img, (30,30),(100,100),(255,0,0), 10)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'Text', (130,130), font, 1, (0,255,255), 2, cv2.LINE_AA)

cv2.imshow('gray', img)
cv2.waitKey(0)
cv2.destroyAllWindows()