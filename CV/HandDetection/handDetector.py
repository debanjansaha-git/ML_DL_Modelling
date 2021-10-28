# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:48:05 2021
@description: This module performs a gesture control on 20 hand landmark points provided part of Mediapipe
              Please check out the documentation of the Mediapipe Hands module in the below URL
              https://google.github.io/mediapipe/solutions/hands.html
              This modular gesture detector can be called by other Gesture Control Projects
@author :     Debanjan Saha
@licence:     MIT License
"""
import cv2
import mediapipe as mp
import time

class handDetector():

    def __init__(self):   #  (mode=False, maxHands=2, DetectConf=0.5, TrackConf=0.5):
        # self.mode = mode
        # self.maxHands = maxHands
        # self.DetectConf = DetectConf
        # self.TrackConf = TrackConf      
        self.model = mp.solutions.hands.Hands() ## (self.mode, self.maxHands, self.DetectConf, self.TrackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.model.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True):       
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255,255,0), cv2.FILLED)       
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
                
        if len(lmList) != 0:
            # display pixel points of gesture point 4
            print(lmList[4])

        # Print FPS on screen
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'FPS: ' + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,0,255), 3)
        
        # Output
        cv2.imshow("Image", img)
        # press keyboard 'q' to quit()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
