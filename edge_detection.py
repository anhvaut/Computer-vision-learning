import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx  = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 5)
    sobely  = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
    edges = cv2.Canny(frame, 100, 200)


    cv2.imshow('frame', frame)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    cv2.imshow('edges', edges)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()