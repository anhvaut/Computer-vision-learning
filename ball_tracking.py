import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    blur = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor( blur, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 100, 150])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange( hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    #res = cv2.bitwise_and(frame, frame, mask = mask)
    

    # kernel =  np.ones((15,15), np.float32) / 255
    # smothed = cv2.filter2D(res, -1, kernel)

    #blur = cv2.GaussianBlur(res, (15, 15), 0)

    #median = cv2.medianBlur(res, 15)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    #cv2.imshow('res', res)
    #cv2.imshow('smothed', smothed)
    #cv2.imshow('blur', blur)
    #cv2.imshow('median', median)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()