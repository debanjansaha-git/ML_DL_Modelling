# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:48:05 2021
@description: Custom Object Detection Module using OpenCV
              Model and weights: SSD MobileNet v3 2020
              https://docs.openvino.ai/latest/omz_models_model_mobilenet_ssd.html
              It detects objects from a set of 91 labels as part of COCO dataset
@author :     Debanjan Saha
@licence:     MIT License
"""
import cv2

thresh = 0.5    # confidence threshold
capture = cv2.VideoCapture(0)
capture.set(3,1080)
capture.set(4,720)

# extract labels into a list
classNames = []
classFile = 'SSD_MobileNet_v3/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# set up SSD MobileNet with OpenCV
configPath = 'SSD_MobileNet_v3/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'SSD_MobileNet_v3/weights_frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# main function
while capture.isOpened():
    readOk, img = capture.read()

    if not readOk:
        break

    # detection
    classIds, confs, bbox = net.detect(img, confThreshold=thresh)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,255,0), thickness=2)
            # display object name
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[0]+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # display object confidence
            cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Output", img)
        # press keyboard 'q' to quit()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()