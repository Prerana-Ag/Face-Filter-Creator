import cv2
import numpy as np
#init camera
cap=cv2.VideoCapture(0)
#face detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_data=[]
skip=0
i=0
j=0
data_path='./Data/'
file_name=input("Enter your Name:")
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    i+=1
    j=0
    print("frame : %d"% i)
    for face in faces:
        j+=1
        print("face: %d "% j)
        (x,y,w,h)=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),color=(255,0,0))
        offset=10
        face_section="face_section_{}".format(j)
        print(face_section)
        face_section=frame[y-offset:y+offset+h,x-offset:x+offset+w]
        face_section=cv2.resize(face_section,(100,100))
        cv2.imshow("Face Section {}".format(j),face_section)
       
        face_section=np.asarray(face_section)
        face_section=face_section.reshape(face_section.shape[0],-1)
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
   
    cv2.imshow("Frame",frame)
    key_pressed=cv2.waitKey(1)&0xff
    if key_pressed==ord('q'):
        break
np.save(data_path+file_name+'.npy',face_data)
print("DAta saved at"+data_path+file_name+'.npy')    
print("video turning off")
cap.release()
cv2.destroyAllWindows()