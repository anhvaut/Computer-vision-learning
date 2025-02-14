import numpy as np
import cv2
import imutils

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

    ball_cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  	
    ball_cnts = imutils.grab_contours(ball_cnts)
    ball_center = None
 
	# only proceed if at least one contour was found
    if len(ball_cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(ball_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# only proceed if the radius meets a minimum size
        if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)
 

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