import cv2
import numpy as np
from PIL import Image
import os
recognizer = cv2.face.LBPHFaceRecognizer_create()
path="datasets"
def getImgageID(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagePaths in imagePath:
        faceImgage=Image.open(imagePaths).convert('L')
        faceNP=np.array(faceImgage)
        id=(os.path.split(imagePaths)[-1].split('.')[1])
        id=int(id)
        ids.append(id)
        faces.append(faceNP)
        cv2.imshow("traning",faceNP)
        cv2.waitKey(1)
    return ids,faces

ids,facedata=getImgageID(path)
recognizer.train(facedata,np.array(ids))
recognizer.write("trainer.yml")
cv2.destroyAllWindows()
print('training completed ')