import time
import numpy as np
from deepface import DeepFace
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# List of models to evaluate
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace", "Dlib", "SFace", "GhostFaceNet"]

# List of distance metrics
distance_metrics = ["cosine", "euclidean", "euclidean_l2"]

# Function to evaluate model prediction time
def evaluate_model(model_name):
    
    # Build face recognition model
    model: FacialRecognition = DeepFace.build_model(model_name=model_name)
    target_size = model.input_shape

    # Measure time taken for prediction
    start_time = time.time()

    # Test different faces
    # Load images and find embeddings
    img1 = DeepFace.extract_faces(img_path="dataset/biden.jpg", target_size=target_size)[0]["face"]
    img1 = np.expand_dims(img1, axis=0)  # to (1, 224, 224, 3)
    img1_representation = model.find_embeddings(img1)

    img2 = DeepFace.extract_faces(img_path="dataset/karen1.jpg", target_size=target_size)[0]["face"]
    img2 = np.expand_dims(img2, axis=0)
    img2_representation = model.find_embeddings(img2)

    img1_representation = np.array(img1_representation)
    img2_representation = np.array(img2_representation)

    # distance between two images - euclidean distance formula
    distance_vector = np.square(img1_representation - img2_representation)
    current_distance = np.sqrt(distance_vector.sum())

    threshold = verification.find_threshold(model_name=model_name, distance_metric="euclidean")

    if current_distance < threshold:
        logger.info("Same")
    else:
        logger.info("Different")

    # Test same faces
    # Load images and find embeddings
    img1 = DeepFace.extract_faces(img_path="dataset/karen.jpg", target_size=target_size)[0]["face"]
    img1 = np.expand_dims(img1, axis=0)  # to (1, 224, 224, 3)
    img1_representation = model.find_embeddings(img1)

    img2 = DeepFace.extract_faces(img_path="dataset/karen1.jpg", target_size=target_size)[0]["face"]
    img2 = np.expand_dims(img2, axis=0)
    img2_representation = model.find_embeddings(img2)

    img1_representation = np.array(img1_representation)
    img2_representation = np.array(img2_representation)

    # distance between two images - euclidean distance formula
    distance_vector = np.square(img1_representation - img2_representation)
    current_distance = np.sqrt(distance_vector.sum())

    threshold = verification.find_threshold(model_name=model_name, distance_metric="euclidean")

    if current_distance < threshold:
        logger.info("Same")
    else:
        logger.info("Different")

    elapsed_time = time.time() - start_time
    logger.info(f"Time taken for predictions with {model_name}: {elapsed_time:.4f} seconds")

# Evaluate each model
for model_name in models:
    evaluate_model(model_name)
