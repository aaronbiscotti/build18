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
import time
from tqdm import tqdm
import gc

friends = [
    {
        "name": "Barack Obama",
        "file_path": "images/obama.jpeg"
    },
    {
        "name": "Aaron",
        "file_path": "images/aaron.jpg"
    },
    {
        "name": "Dylan",
        "file_path": "images/dylan.jpeg"
    },
    {
        "name": "Gina",
        "file_path": "images/gina.jpeg"
    },
    {
        "name": "Gio",
        "file_path": "images/gio.jpeg"
    },
    {
        "name": "Justin",
        "file_path": "images/justin.jpeg"
    }
]

def capture_image():
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image_file:
            capture_command = ['libcamera-still', '-o', temp_image_file.name]
            subprocess.run(capture_command, check=True)
            return temp_image_file.name
    except Exception as e:
        print(f'Error capturing image: {e}')

# Load and encode faces from friends list
def load_known_faces(friends):
    known_face_encodings = []
    known_face_names = []
    for friend in tqdm(friends):
        # Load each friend's image
        image = face_recognition.load_image_file(friend["file_path"])

        # Attempt to find face encodings in the image
        encodings = face_recognition.face_encodings(image)

        # Check if at least one face was found
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(friend["name"])
        else:
            print(f"No faces found in {friend['file_path']}. Skipping this image.")
    return known_face_encodings, known_face_names

def main():
    print("Loading face encodings...")
    known_face_encodings, known_face_names = load_known_faces(friends)

    while True:
        print(f"Capturing image...")
        image_path = capture_image()
        image = cv2.imread(image_path)

        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if not face_encodings:
            print(f"No face encodings found")
            continue

        print(f"Running face-matching...")
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = known_face_names[best_match_index]

            print(f"Recognized: {name}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        with open("recognized_face.txt", "w") as file:
            file.write(name)

        gc.collect()
        time.sleep(5)

if __name__ == "__main__":
    main()
