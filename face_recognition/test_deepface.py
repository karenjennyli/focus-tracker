import time
import numpy as np
from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition

# build face recognition model
model_name = "SFace"
model: FacialRecognition = DeepFace.build_model(model_name=model_name)
target_size = model.input_shape

# load images and find embeddings
start_time = time.time()
img1 = DeepFace.extract_faces(img_path="dataset/karen.jpg", target_size=target_size, detector_backend="ssd")[0]["face"]
print("Time taken to extract face 1:", time.time() - start_time)
img1 = np.expand_dims(img1, axis=0)
karen_embeddings = model.find_embeddings(img1)
karen_embeddings = np.array(karen_embeddings)

img2 = DeepFace.extract_faces(img_path="dataset/mom.jpg", target_size=target_size)[0]["face"]
img2 = np.expand_dims(img2, axis=0)
mom_embeddings = model.find_embeddings(img2)
mom_embeddings = np.array(mom_embeddings)

img3 = DeepFace.extract_faces(img_path="dataset/mom_karen.jpg", target_size=target_size)
