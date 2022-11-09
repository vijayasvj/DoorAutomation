from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import face_recognition
import streamlit as st
import numpy as np
import yaml

with open(r'YAML encodings/emp_face_encodings.yml', 'r') as f:
    emp_face_encodings = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_face_names.yml', 'r') as f:
    emp_face_names = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_phno.yml', 'r') as f:
    emp_phno = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_id.yml', 'r') as f:
    emp_id = yaml.load(f.read(), Loader=yaml.Loader)


class VideoProcessor:
	def recv(self, frame):
		imggg = frame.to_ndarray(format="bgr24")
		testing = imggg
		face_locations = face_recognition.face_locations(testing)
		face_encodings = face_recognition.face_encodings(testing, face_locations)

		# Loop through each face in this frame of video
		for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(emp_face_encodings, face_encoding)

			name = "Unknown"
			phno = "Unknown"
			empid = "Unknown"
			
			# If a match was found in known_face_encodings, just use the first one.
			# if True in matches:
			#     first_match_index = matches.index(True)
			#     name = known_face_names[first_match_index]

			# Or instead, use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(emp_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = emp_face_names[best_match_index]
				phno = emp_phno[best_match_index]
				empid = emp_id[best_match_index]

			# Draw a box around the face
			cv2.rectangle(testing, (left, top), (right, bottom), (0, 0, 255), 2)

			# Draw a label with a name below the face
			cv2.rectangle(testing, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(testing, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
			

		return av.VideoFrame.from_ndarray(imggg, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)

