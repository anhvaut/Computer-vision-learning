import cv2
import numpy as np

image = cv2.imread("pictures/hd2.jpg")

height, width = image.shape[:2]
scaledWidth = 900
scaledHeight = int((scaledWidth * height) / width)
image = cv2.resize(image, (scaledWidth, scaledHeight), fx= 0.5, fy=0.5, interpolation= cv2.INTER_AREA)

im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray,(5,5),2)


ret, im_th = cv2.threshold(im_gray,94,200,cv2.THRESH_BINARY_INV)
ctrs,_ = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects =[cv2.boundingRect(ctr) for ctr in ctrs]

# height, width = image.shape[:2]
# scaledWidth = 900
# scaledHeight = int((scaledWidth * height) / width)
# image = cv2.resize(image, (scaledWidth, scaledHeight), fx= 0.5, fy=0.5, interpolation= cv2.INTER_AREA)
# im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# im_gray = cv2.GaussianBlur(im_gray, (5,5), 0)

# cv2.imshow('gray', im_gray)

# kernel = np.ones((3,3),dtype='uint8')
# ret, im_th = cv2.threshold(im_gray, 115, 255, cv2.THRESH_BINARY_INV)
# canny = cv2.Canny(im_th, 70, 170)
# # canny = cv2.dilate(canny, kernel, iterations = 1)
# ctrs = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# new_rects = []
# for rect in rects:
#    if rect[2] > 10 and rect[3] > 20 and rect[3] < 100 and not check_contain_another_rect(rect, rects):
#        new_rects.append(rect)

# new_rects = np.array(new_rects)

# for rect in new_rects:
#     leng = int(rect[3] * 1.6)
#     pt1 = int(rect[1] + rect[3]/2 - leng/2)
#     pt2 = int(rect[0] + rect[2]/2 - leng/2)
#     roi = im_th[pt1: pt1 + leng, pt2: pt2+leng]
#     roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
#     number = np.array([roi]).reshape(1, (28*28))
#     ret = model.predict(number)
#     cv2.rectangle(im_gray, (pt2, pt1), (pt2 + leng, pt1+leng), (0, 255, 0), 1)
#     cv2.putText(im_gray, str(int(ret[0])), (pt2, pt1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)



# cv2.imshow(im_th)

for rect in rects:
    # if (rect[2])>130 and (rect[3])>130:
    cv2.rectangle(image,(rect[0],rect[1]),(rect[2]+rect[0],rect[3]+rect[1]),(0,255,0),15)
    len = int(rect[3]*1.6)
    pt1 = int(rect[1]+rect[3]//2-len//2)
    pt2 = int(rect[0]+rect[2]//2-len//2)
    roi = im_th[pt1:pt1+len,pt2:pt2+len]
        # roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)
        # roi = cv2.dilate(roi,(3,3))
        # roi_r =np.array([roi]).reshape(1,28*28)
        
    print (roi)
        # ret = model.predict(roi_r)
       # print ret
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(image,str(int(ret[0])),(rect[0],rect[1]), font, 10, (0,255,0),10,cv2.FONT_HERSHEY_SIMPLEX)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()