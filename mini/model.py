import cv2
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
namelist=["","sumedh","mam","bharati","akka","aditya","varshini"]
imgBackground=cv2.imread("background.jpeg")
while True:

    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        serial,conf=recognizer.predict(gray[y:y+h,x:x+w])
        if conf<50:
             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
             cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
             cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
             cv2.putText(frame,namelist[serial],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(225,225,255),2)
             cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame,"unknown",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(225,225,255),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,225),1)
    imgBackground[190:190+480,150:150+640]=frame
    cv2.imshow("Frame",imgBackground)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
print("detaset collection complete")