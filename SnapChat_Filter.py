import cv2
import numpy as np
img=cv2.imread("./Face Recognization Files/SnapChat_Filter_Challenge/Train/Jamie_Before.jpg")
glass=cv2.imread("./Face Recognization Files/SnapChat_Filter_Challenge/Train/glasses.png")
mustache=cv2.imread("./Face Recognization Files/SnapChat_Filter_Challenge/Train/mustache.png")
img=cv2.resize(img,(500,500))

eye_cascade=cv2.CascadeClassifier("./Face Recognization Files/frontalEyes35x16.xml")
nose_cascade=cv2.CascadeClassifier("./Face Recognization Files/Nose18x15.xml")
eyes=eye_cascade.detectMultiScale(img,1.3,9)
noses=nose_cascade.detectMultiScale(img,1.3,8)
for eye in eyes:
    (x,y,w,h)=eye
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    glass=cv2.resize(glass,(w,h))
    img[y:y+h,x:x+w]=glass
    cv2.imshow("glass",glass)

for nose in noses:
    (x,y,w,h)=nose
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    mustache=cv2.resize(mustache,(w,h))
    img[y:y+h,x:x+w]=mustache
    cv2.imshow("m",mustache)


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

