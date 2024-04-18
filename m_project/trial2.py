import os
import face_recognition

# Load the target image
target_image_path = "m.jpg"
target_image = face_recognition.load_image_file(target_image_path)
target_encoding = face_recognition.face_encodings(target_image)[0]

# Directory containing the images to compare against
compare_dir = "test"
for filename in os.listdir(compare_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        image_path = os.path.join(compare_dir, filename)
        compare_image = face_recognition.load_image_file(image_path)

        # Find face encodings for the compare image
        compare_encodings = face_recognition.face_encodings(compare_image)

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
        else:
            print(f"{filename}: The face belongs to a different person.")
