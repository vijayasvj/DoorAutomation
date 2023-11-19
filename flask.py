from flask import Flask, render_template, Response
import cv2
import requests
import numpy as np
import av
import face_recognition
import numpy as np
import yaml
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime

import os

csv_file = '/Users/apple/Documents/SKPC/DoorAutomation/data.csv'

timenow = str(datetime.datetime.now().strftime('%H'))

# Check if the CSV file exists, and create it with headers if not
if not os.path.exists(csv_file):
    with open(csv_file, 'w') as file:
        file.write("ID,time\n")

# A set to keep track of seen IDs
seen_ids = set()


with open(r'emp_face_encodings.yml', 'r') as f:
    emp_face_encodings = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'emp_face_names.yml', 'r') as f:
    emp_face_names = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'emp_phno.yml', 'r') as f:
    emp_phno = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'emp_id.yml', 'r') as f:
    emp_id = yaml.load(f.read(), Loader=yaml.Loader)


# Function to add a new entry to the CSV file
def add_to_csv(id, time, mail):
    if id not in seen_ids:
        with open(csv_file, 'a') as file:
            file.write(f"{id},{time}\n")
            seen_ids.add(id)
        datenow = str(datetime.datetime.now().strftime("%d/%m/%y"))
        subject = "Attendance for " + datenow + ", at " + time
        body = "Your attendance for today have been added to the database"
        sender_email = "skpc.adrig@gmail.com"
        receiver_email = mail
        password = "kygyemxsfwlkwoqq"
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        text = message.as_string()
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)
app = Flask(__name__)

# Replace this with the actual ESP32-CAM URL
esp32_cam_url = "http://admin:123456@192.168.0.109/cgi-bin/snapshot.cgi?chn=0&u=admin&p=123456"

# Function to fetch frames from the ESP32-CAM
def gen_frames():
    while True:
        response = requests.get(esp32_cam_url)
        frame = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(frame, -1)
        testing = frame
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
            print(face_distances)
            best_match_index = np.argmin(face_distances)
            if(face_distances[best_match_index]<0.5):
                if matches[best_match_index]:
                    name = emp_face_names[best_match_index]
                    phno = emp_phno[best_match_index]
                    empid = emp_id[best_match_index]
                    timenow = str(datetime.datetime.now().strftime('%H:%M'))
                    add_to_csv(empid, timenow, phno)
                # Draw a box around the face
            cv2.rectangle(testing, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(testing, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(testing, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        frame = testing 
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
