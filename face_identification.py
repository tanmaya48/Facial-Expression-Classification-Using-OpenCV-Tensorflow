import cv2
import numpy as np


from tensorflow.keras.models import model_from_json


####  Loading tensorflow model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#####



cap = cv2.VideoCapture(0)


### Loading HAAR Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

smile = 0

while True:

    _,img = cap.read()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  ## finding faces in image


    
    for (x,y,w,h) in faces:  ## We are assuming only one face is present in the image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]

        roi_gray = cv2.resize(roi_gray, (100, 100))

        face = roi_gray[:,:,np.newaxis]   ## converting image data into proper shape for tensorflow model

        face = np.expand_dims(face, axis=0) 
    
        smile = loaded_model.predict(face)[0,0]  ## check face image with model
    
    if smile >= 0.8:
        cv2.putText(img,'Smiling :)',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,)


    cv2.imshow('game',img)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cap.release() 
cv2.destroyAllWindows()    
