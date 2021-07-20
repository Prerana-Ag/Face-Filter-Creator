import cv2
import numpy as np
import os
#init camera
cap=cv2.VideoCapture(0)
#face detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
data_path="./Data/"
############################# KNN  #########################
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(X,Y,query_point,k=50):
    vals=[]
    m=X.shape[0]
    for i in range(m):
        d=dist(query_point,X[i])
        vals.append((d,Y[i]))
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals, dtype=object)
    new_vals=np.unique(vals[:,1],return_counts=True)
    index=np.argmax(new_vals[1])
    prediction=new_vals[0][index]
    return prediction
###################################################
class_id=0
face_data=[]
name=[]
labels=[]
# Loading Data
for fx in os.listdir(data_path):
    if fx.endswith('.npy'):
        print(fx)
        data_item=np.load(data_path+fx)
        print(data_item.shape)
        face_data.append(data_item)
        name.append(fx[:-4])
        #creating labels
        target=class_id*np.ones(data_item.shape[0],)
        labels.append(target)
        class_id+=1
X_train=np.concatenate(face_data,axis=0)
Y_train=np.concatenate(labels,axis=0)
X_train=X_train.reshape(X_train.shape[0],-1)
Y_train=Y_train.reshape(Y_train.shape[0],-1)
print(X_train.shape)       
print(Y_train.shape) 
print(name)
        
#########################################
i=0
skip=0
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
        skip+=1
        out=knn(X_train,Y_train,face_section.flatten())
        pred_name=name[int(out)]
        print(pred_name)
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        print(skip)
    cv2.imshow("Frame",frame)
    key_pressed=cv2.waitKey(1)&0xff
    if key_pressed==ord('q'):
        break
   
print("video turning off")
cap.release()
cv2.destroyAllWindows()