# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 07:39:47 2021
@description: This module allows you to scan documents
@author :     Debanjan Saha
@licence:     MIT License
"""

import numpy as np
import cv2

width = 480
height = 640
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cap.set(10,130)

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 200,200)

    kernel = np.ones((5,5))
    imgDilat1 = cv2.dilate(imgCanny, kernel, iterations=2)
    imgDilat2 = cv2.dilate(imgDilat1, kernel, iterations=2)
    imgThresh = cv2.erode(imgDilat2, kernel, iterations=1)
#    print(imgThresh)
    return imgThresh

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    addPts = myPoints.sum(1)
    # find the top left and bottom right corners
    myPointsNew[0] = myPoints[np.argmin(addPts)]
    myPointsNew[3] = myPoints[np.argmax(addPts)]
    # find the other two points
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    return imgOutput

while True:
    success, img = cap.read()
    img = cv2.resize(img, (width, height))
    imgContour = img.copy()
    imgTh = preProcessing(img)
    biggest = getContours(imgTh)
#    print(biggest)
    imgWarped = getWarp(img, biggest)

    cv2.imshow("Result", imgWarped)
    # press keyboard 'q' to quit()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break