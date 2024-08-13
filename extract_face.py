import cv2
import matplotlib.pyplot as plt
from glob import glob
import os

face_cascade = cv2.CascadeClassifier(r'model/haarcascade_frontalface_alt.xml')

def detectAndDisplay(frame, name):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    
    faces = face_cascade.detectMultiScale(frame_gray)

    for i, (x, y, w, h) in enumerate(faces):
        
        faceROI = frame[y:y + h, x:x + w]
        faceROI_size = cv2.resize(faceROI, (256, 256), interpolation = cv2.INTER_LINEAR)
        filename = f'face_{name}.jpg'

        cv2.imwrite(os.path.join('all_faces' , filename), faceROI_size)

# Loop through the images in the 'photo' directory
photo_files = glob('photo/*')
for i, img_path in enumerate(photo_files):
    name = img_path.split('\\')[-1].split('.')[-2][:-1]
    image = cv2.imread(img_path)
    detectAndDisplay(image, name)
