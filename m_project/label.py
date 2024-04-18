import dlib
import numpy as np
import os

# Load the pre-trained face detection and recognition models
detector = dlib.get_frontal_face_detector()
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Path to the directory containing images of known faces
known_faces_dir = "known_faces"

# Initialize lists to store face embeddings and corresponding labels
known_face_embeddings = []
known_face_labels = []

# Iterate over the images of known faces
for file_name in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, file_name)
    # Load the image
    img = dlib.load_rgb_image(image_path)
    # Detect faces in the image
    faces = detector(img)
    # Iterate over detected faces
    for face in faces:
        # Compute face embeddings
        face_embedding = np.array(face_rec_model.compute_face_descriptor(img, face))
        # Append the embedding to the list of known face embeddings
        known_face_embeddings.append(face_embedding)
        # Extract the label (name or ID) from the file name
        label = file_name.split(".")[0]  # Assuming file names are in the format "label.jpg"
        # Append the label to the list of known face labels
        known_face_labels.append(label)

# Convert lists to NumPy arrays for efficiency
known_face_embeddings = np.array(known_face_embeddings)
known_face_labels = np.array(known_face_labels)

# Save the face embeddings and labels to files for later use
np.save("known_face_embeddings.npy", known_face_embeddings)
np.save("known_face_labels.npy", known_face_labels)
