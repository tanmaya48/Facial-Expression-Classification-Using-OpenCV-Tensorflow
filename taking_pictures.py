import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load HAAR cascade from where it is stored in your directory

i = 0

while True:

    _,img = cap.read()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        cv2.imwrite('image'+str(i)+'.png',roi_gray)   # save expression image
        i+=1
        time.sleep(0.2)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release() 
cv2.destroyAllWindows()    