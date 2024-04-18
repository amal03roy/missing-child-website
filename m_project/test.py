from deepface import DeepFace

f1 = "anto.jpg"
f2 = "mohanlal_trial.jpg"
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
result = DeepFace.verify(img1_path=f1, img2_path=f2, detector_backend=backends[1])
