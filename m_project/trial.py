import os
import face_recognition
import numpy as np
import cv2

# Load the target image
#sharpen_filter=np.array([[-1,-1,-1],
#                 [-1,9,-1],
               # [-1,-1,-1]])

#original= cv2.imread('anto.jpg', cv2.IMREAD_UNCHANGED)
#sharp_image=cv2.filter2D(original,-1,sharpen_filter)
#cv2.imwrite('anto.jpg', sharp_image)




target_image_path = "m.jpg"
target_image = face_recognition.load_image_file(target_image_path)
target_encoding = face_recognition.face_encodings(target_image)[0]

# Directory containing the images to compare against
compare_dir = "testdir"
for filename in os.listdir(compare_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        compare_image = face_recognition.load_image_file(f"{compare_dir}\\{filename}")

        # Find face encodings for the compare image
        compare_encodings = face_recognition.face_encodings(compare_image)

        # results = face_recognition.compare_faces([target_encoding], compare_encodings, tolerance=0.6)
        # print(results)

        # If no face is found, skip
        if len(compare_encodings) == 0:
            print(f"No face detected in {filename}.")
            continue

        # Compare face encoding of the target image with the compare image
        compare_encoding = compare_encodings[0]
        face_distance = face_recognition.face_distance([target_encoding], compare_encoding)[0]

        # Define a threshold for face similarity
        threshold = 0.6  # Adjust as needed

        # Compare face distance to threshold
        if face_distance < threshold:
            print(f"{filename}: The face belongs to the same person.")
            break
        else:
            print(f"{filename}: The face belongs to a different person.")
#Activate  virtual env
# C:\Users\LEGION\OneDrive\Desktop\m_project\.venv\Scripts\activate.bat
