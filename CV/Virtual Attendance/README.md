# Face Recognition Attendance System

## Introduction

This module is a face recognition and attendance tracking tool that detects faces from live webcam and records attendance in a csv file if a known face is found, which can be used to load the data into a database. 
The module uses the `face_recognition` library to perform face detection and recognition. The webcam input is captured using the cv2 library.


## Getting Started

The following dependencies need to be installed before running the code:

`numpy`
`cv2`
`face_recognition`
`os`
`datetime`

## Execute Program

To run the code, simply execute the script in a Python environment. The script will open a window showing the webcam feed and will begin detecting and recognizing faces.

## Database

The script uses a directory called 'KnownDB' to store images of known faces for comparison. The name of the image files are used as the name of the person for attendance tracking. The script reads all images from this directory and encodes them for comparison with webcam feed.

## Attendance Tracking

The script records attendance in a csv file called 'attendance.csv'. Each row of the file contains the name of the person and the time of attendance. If a known face is detected, the script checks if the person's name is already in the attendance file. If not, the script adds a new row to the file with the person's name and the current time.
This csv file is then used to load the records into any cloud database table which can be used to create a view to display the results in a dashboard.

## Functionality

The script captures webcam feed and resizes it for faster processing. 
It then detects faces in the current frame using the `face_locations` function from the `face_recognition` library. 
The script then encodes the detected faces for comparison with known faces using the `face_encodings` function.
The script compares the encoded faces with the known faces using the `compare_faces` function and calculates the distance between the faces using the `face_distance` function. 
The script then finds the closest match and verifies if it is a known face. 
If a match is found, the name of the person is determined using the classnames list and the attendance is recorded in a csv file using the `markAttendance` function. 
The recognized face is then highlighted on the video feed with a rectangle and the person's name is displayed on top of the rectangle. 
The code runs in a continuous loop until the 'q' key is pressed, which breaks the loop and exits the program.

## Usage

   1. Place the images of known faces in a folder named "KnownDB" in the same directory as the script.
   2. Run the script using `python3 script_name.py`
   3. Allow the script to access the webcam when prompted
   4. The script will run in a loop and display the video feed from the webcam on the screen. If a known face is detected, the name of the person will be displayed on the screen and recorded in the "attendance.csv" file.
   5. Press 'q' on the keyboard to quit the script.

## Customization

The scaleFactor variable can be adjusted to change the size of the video feed displayed on the screen.
The format of the date and time recorded in the "attendance.csv" file can be changed in the markAttendance function.
Additional functionality can be added to the script by modifying the markAttendance function.

## Limitations
Some limitations of this module include:

- The module only detects one face per frame, thus it may not accurately record attendance if multiple faces are present in the frame.
- The module relies on accurate encoding of known faces, thus if the images of known faces in the "KnownDB" folder are of low quality or do not accurately represent the person, the module may not correctly match and record attendance.
- The module requires a webcam to function and cannot be used with pre-recorded video or images.
- The module uses the 'attendance.csv' file to record attendance, if the file is not present or is not in the correct format, the module will not function properly.
- The module uses the datetime library to record the time of attendance, it may not work properly if the system time is not accurate.
- If a new user wants to be added, their images should be added to the 'KnownDB' folder and the module should be run again to include the new user.
- The module is not resistant to spoofing/fake faces.
- The module can only detect people in the knownDB folder, it is not able to detect unknown people.
