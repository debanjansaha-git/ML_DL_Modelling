# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 07:39:47 2021
@description: This module allows you to draw on your screen using OpenCV
              In order to add more colors please add the respective colour in myColorValues
              Also, use the create_mask.py to create a mask for that color and add to myColors
@author :     Debanjan Saha
@licence:     MIT License
"""

import numpy as np
import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,130)



myColors = [[84,154,162,156,252,255], # blue
            [54,52,95,91,214,214],    # green
            [25,46,221,103,127,255]   # fluroscent
            ]

myColorValues = [
                    [255,178,102],      # blue
                    [0,153,0],          # green
                    [102,255,178]       # fluroscent
                ]

myPoints = [] # [x, y, colorID]

def findColor(img, myColors,myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    colorPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        cv2.circle(imgRes,(x, y), 10, myColorValues[count], cv2.FILLED)
        # cv2.imshow(str(color[0]), mask)
        if x!= 0 and y!=0:
            colorPoints.append([x, y, count])
        count += 1
    return colorPoints


def getContours(img):
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(imgRes, cnt, -1, (255,0,0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y

def drawOnCanvas(myPoints, myColorValues):
    for points in myPoints:
        cv2.circle(imgRes,(points[0], points[1]), 10, myColorValues[points[2]], cv2.FILLED)

while True:
    success, img = cap.read()
    imgRes = img.copy()
    newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints) != 0:
        for newPts in newPoints:
            myPoints.append(newPts)
        drawOnCanvas(myPoints, myColorValues)
        
    cv2.imshow("Result", imgRes)
    # press keyboard 'q' to quit()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
