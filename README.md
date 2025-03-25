# Face Recognition System

## Overview
This project implements a face recognition system using Haar cascade classifiers for face detection and the DeepFace library for face recognition. It provides tools for detecting faces in images or video streams and matching them against a database of known faces.

## Features
- Face detection using Haar cascade classifiers
- Eye detection with support for eyeglasses
- Real-time face recognition in video streams
- Support for matching faces against a database

## Prerequisites
- Python 3.6+
- OpenCV
- DeepFace
- TensorFlow (required by DeepFace)
- NumPy

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/face_recognition.git
cd face_recognition
```

2. Install required packages:
```
pip install opencv-python deepface tensorflow numpy
```

## Project Structure
```
face_recognition/
├── model/
│   ├── haarcascade_frontalface_alt.xml  # Face detection model
│   └── haarcascade_eye_tree_eyeglasses.xml  # Eye detection model
├── recognition_demo.ipynb  # Jupyter notebook with demo code
└── README.md  # This file
```

## Usage

### Running the Demo

You can run the recognition demo using the provided Jupyter notebook:

```python
from deepface import DeepFace

# Stream video and perform face recognition
DeepFace.stream(db_path="path/to/your/database")
```

### Creating Your Own Database

1. Create a folder to store face images
2. Add face images to the folder with appropriate naming
3. Use this folder path as the `db_path` parameter

## License
This project uses pre-trained models from OpenCV which are licensed under the Intel License Agreement for Open Source Computer Vision Library.

## Acknowledgements
- OpenCV for providing Haar cascade classifiers
- DeepFace team for the face recognition library