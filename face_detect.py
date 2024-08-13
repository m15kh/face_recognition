from deepface import DeepFace
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
image1 = cv.imread('photo/1.jpg')
image1 =  cv.cvtColor(image1, cv.COLOR_BGR2RGB)

# Display image


result = DeepFace.verify(
    img1_path="photo/3.jpg",
    img2_path="photo/2.jpg",
    threshold=0.5
)

print(result)
face1 = result['facial_areas']['img1']
print(face1)
x = int(face1['x'])
y = int(face1['y'])
w = int(face1['w'])
h = int(face1['h'])

# Draw rectangle
# image1 = cv.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 4)

# image1_array = np.array(image1)
# plt.imshow(image1)
# plt.show()
print(result["verified"])