"""
Date:   03-Nov-2021
Author: Debanjan
Description:
This module is used to detect face and eye features from images and webcam
It uses OpenCV Haar Cascade Trained Models for detection
The trained models can be found in the below URL
https://github.com/opencv/opencv/tree/master/data/haarcascades

"""
import os
from glob import glob
import numpy as np
import cv2

# Load Models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./models/haarcascade_eye.xml')

def detect_face_eyes(detect_from_image=True):
    
    # Detect face and eyes from images
    if detect_from_image:
        img_path = [os.path.relpath(x) for x in glob(os.getcwd() + '/*/*.jpg')]
        for image in img_path:
            img = cv2.imread(image)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Multiscale refers to detecting objects (faces) at multiple scales. 
            faces = face_cascade.detectMultiScale(gray_img, 1.3, 5) # scaleFactor = 1.3, minNeighbors = 3
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

                # Define Region of Interest which is the cropped face
                roi_gray = gray_img[y:y+h, x:x+w]   # Crop only face from gray image
                roi_color = img[y:y+h, x:x+w]       # Crop only face from color image for display purpose

                # Define eye classifier and use gray image to detect the eyes
                eyes = eye_cascade.detectMultiScale(roi_gray) 
                for (ex,ey,ew,eh) in eyes:
                    # Draw blue bounding boxes around the eyes
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2) 

            cv2.imshow('Image', img)
            cv2.waitKey(0)

    # Use live webcam to detect features
    else:
        capture = cv2.VideoCapture(0)
        while True:
            _, img = capture.read()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Multiscale refers to detecting objects (faces) at multiple scales. 
            faces = face_cascade.detectMultiScale(gray_img, 1.3, 5) # scaleFactor = 1.3, minNeighbors = 3
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

                # Define Region of Interest which is the cropped face
                roi_gray = gray_img[y:y+h, x:x+w]   # Crop only face from gray scale
                roi_color = img[y:y+h, x:x+w]       # Crop only face from color scale for display purpose

                # Define eye classifier and use gray image to detect the eyes
                eyes = eye_cascade.detectMultiScale(roi_gray) 
                for (ex,ey,ew,eh) in eyes:
                    # Draw blue bounding boxes around the eyes
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2) 

            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face_eyes(False)