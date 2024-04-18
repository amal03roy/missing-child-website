import os
import face_recognition

# Load the target image
target_image_path = "m.jpg"
target_image = face_recognition.load_image_file(target_image_path)
target_encoding = face_recognition.face_encodings(target_image)

# Directory containing the images to compare against
compare_dir = "testdir"
threshold = 0.6  # Adjust as needed
similar_face_found = False

# Iterate through all images in the directory
for filename in os.listdir(compare_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        compare_image = face_recognition.load_image_file(os.path.join(compare_dir, filename))

        # Find face encodings for the compare image
        compare_encodings = face_recognition.face_encodings(compare_image)

        # Check if any face is detected in the compare image
        if len(compare_encodings) == 0:
            print(f"No face detected in {filename}.")
            continue

        # Compare face encoding of the target image with the compare image
        for compare_encoding in compare_encodings:
            face_distance = face_recognition.face_distance(target_encoding, compare_encoding)
            if any(face_distance < threshold):
                print(f"{filename}: A similar face was found.")
                similar_face_found = True
                break

if not similar_face_found:
    print("No similar face found in the directory.")
