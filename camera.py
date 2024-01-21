# -*- coding: utf-8 -*-
#
#  camera.py
#  
#  Copyright 2024  <bass@bipolarfish>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import cv2
import subprocess
import numpy as np
import face_recognition
import tempfile
import os

def capture_image():
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image_file:
            capture_command = ['libcamera-still', '-o', temp_image_file.name]
            subprocess.run(capture_command, check=True)
            return temp_image_file.name
    except Exception as e:
        print(f'Error capturing image: {e}')
names = [
    {
        "name": "Barack Obama",
        "file": "obama.jpeg",
    },
    # Add more people here...
]

known_face_encodings = []
for name in names:
    corr_image = face_recognition.load_image_file(name["file"])
    known_face_encodings.append(face_recognition.face_encodings(corr_image)[0])

known_face_names = [
    "Barack Obama",
    # Add more names here...
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    image_path = capture_image()
    image = cv2.imread(image_path)

    if process_this_frame:
        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if name == "Unknown":
            new_name = input("I don't recognize you, what's your name? ")
            screenshot_filename = f"{new_name}.jpg"
            cv2.imwrite(screenshot_filename, image)

            new_face_image = face_recognition.load_image_file(screenshot_filename)
            new_face_encodings = face_recognition.face_encodings(new_face_image)

            if new_face_encodings:
                new_face_encoding = new_face_encodings[0]
                known_face_encodings.append(new_face_encoding)
                known_face_names.append(new_name)
                face_names[-1] = new_name
            else:
                print(f"No face found in the screenshot for {new_name}. Please try again.")

    cv2.imshow('Video', image)
    os.remove(image_path)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


