import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)

    lower = np.array([80, 80, 80])
    upper = np.array([180, 255, 255])

    mask = cv2.inRange( hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode (mask, kernel, iterations = 1)
    dilation = cv2.dilate (mask, kernel, iterations = 1)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('open', opening)
    cv2.imshow('close', closing)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()