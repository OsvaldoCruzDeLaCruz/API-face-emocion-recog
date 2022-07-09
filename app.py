import shutil
import face_recognition
import PIL.Image
import PIL.ImageDraw

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
app = FastAPI()


# image = cv2.imread("sad2.jpg")
# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# plt.show()

@app.get('/')
def index():
    return {"message": "API face recognition emotions"}


@app.post('/recognize')
async def recognize(file: UploadFile = File(...)):
    
    with open(f"{file.filename}", 'wb') as f:
        shutil.copyfileobj(file.file, f)
        

        image = cv2.imread(file.filename)

        results = DeepFace.analyze(image, enforce_detection = False)
        
        print(results)
        print("********************************************************")
        print(results['dominant_emotion'])
        print("********************************************************")

        return {"Emocion": results['dominant_emotion']}




        
