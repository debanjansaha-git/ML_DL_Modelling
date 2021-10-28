from hand_tracker import hand_tracker

import cv2
import mediapipe as md
import time
from math import sqrt
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



class volume_controller:
	def __init__(self):
		self.image_handler = cv2.VideoCapture(0)
		self.HandTracker = hand_tracker()
		
		self.max_value = 200
		self.min_value = 17



	def take_picture(self):
		status, image = self.image_handler.read()

		if status:
			return image

	def change_volume(self, percentage):
		devices = AudioUtilities.GetSpeakers()
		interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
		volume = cast(interface, POINTER(IAudioEndpointVolume))

		value = -65.25 * (1 - percentage)

		if value <= 0: 
			volume.SetMasterVolumeLevel(int(value), None)



	def process_image(self, frame):
		landmarks, results = self.HandTracker.find_hand_positions(frame)#, hand_number=0)
		processed_frame = self.HandTracker.draw_landmarks(frame, results)

		h, w, c = frame.shape

		if len(landmarks) > 0:
			thumb_tip, index_tip = landmarks[4][1], landmarks[8][1]
			thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
			index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

			cv2.line(processed_frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 10)
			cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 0, 255), 15)
			cv2.circle(frame, (index_x, index_y), 5, (0, 0, 255), 15)


			dx, dy = thumb_x - index_x, thumb_y - index_y
			distance = sqrt((dy ** 2) + (dx ** 2))

			if distance > self.max_value:
				percentage = 1
			elif distance < self.min_value:
				percentage = 0
			else:
				percentage = (distance - self.min_value) / (self.max_value - self.min_value)

			self.change_volume(percentage)
		return processed_frame




if __name__ == '__main__':
	VolumeController = volume_controller()
	run = True 

	while run:
		frame = VolumeController.take_picture()
		processed = VolumeController.process_image(frame)

		cv2.imshow('Volume Controller', processed)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			run = False