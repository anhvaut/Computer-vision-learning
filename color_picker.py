fimport cv2
import numpy as np

cap = cv2.VideoCapture(0)


def nothing(arg): pass


# takes an image, and a lower and upper bound
# returns only the parts of the image in bounds
def only_color(frame, (b, r, g, b1, r1, g1), morph):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower = np.array([b, r, g])
    upper = np.array([b1, r1, g1])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)
    # define kernel size (for touching up the image)
    kernel = np.ones((morph, morph), np.uint8)
    # touch up
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res, mask


# setup trackbars
cv2.namedWindow('image')
cv2.createTrackbar('h', 'image', 0, 255, nothing)
cv2.createTrackbar('s', 'image', 0, 255, nothing)
cv2.createTrackbar('v', 'image', 0, 255, nothing)
cv2.createTrackbar('h1', 'image', 0, 255, nothing)
cv2.createTrackbar('s1', 'image', 0, 255, nothing)
cv2.createTrackbar('v1', 'image', 0, 255, nothing)
cv2.createTrackbar('morph', 'image', 0, 10, nothing)

# main loop of the program
while True:

    # read image from the video
    _, img = cap.read()

    # get trackbar values
    h = cv2.getTrackbarPos('h', 'image')
    s = cv2.getTrackbarPos('s', 'image')
    v = cv2.getTrackbarPos('v', 'image')
    h1 = cv2.getTrackbarPos('h1', 'image')
    s1 = cv2.getTrackbarPos('s1', 'image')
    v1 = cv2.getTrackbarPos('v1', 'image')
    morph = cv2.getTrackbarPos('morph', 'image')

    # extract only the colors between h,s,v and h1,s1,v1
    img, mask = only_color(img, (h, s, v, h1, s1, v1), morph)

    # show the image and wait
    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if k == 27: break

# release the video to avoid memory leaks, and close the window
cap.release()
cv2.destroyAllWindows()