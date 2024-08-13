import cv2
import os
from  glob import glob






face_cascade = cv2.CascadeClassifier(r'model/haarcascade_frontalface_alt.xml')

def detectAndSave(frame, name, index):
    if not os.path.exists('./database'):
        os.makedirs('./database')

    person_dir = os.path.join('database', name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for  (x, y, w, h) in (faces):
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (256, 256))
        face_filename = f"{name}_face{index}.jpg"
        face_path = os.path.join(person_dir, face_filename)
        cv2.imwrite(face_path, face_roi)


for img_path in glob('photo/*'):
    full_id = img_path.split('\\')[-1].split('.')[-2]
    name, index = full_id[:-1], full_id[-1]
    image = cv2.imread(img_path)
    detectAndSave(image, name, index)
    
