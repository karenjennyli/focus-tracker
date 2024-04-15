import sys
import cv2
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from deepface import DeepFace
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition

from utils import FACE_DETECTOR_BACKEND, FACE_RECOGNITION_MODEL_NAME, FACE_DISTANCE_METRIC, FACE_RECOGNITION_KEYPOINTS

class FaceRecognizer:

    def __init__(self, width: int, height: int, history_length: int = 20, template_img_path: str = 'calibration_data/template_face.jpg'):
        self.img_width = width
        self.img_height = height
        self.history_length = history_length
        self.history = [True] * history_length
        self.user_recognized = True
        self.user_left_time = None
        self.recognition_model: FacialRecognition = DeepFace.build_model(model_name=FACE_RECOGNITION_MODEL_NAME)
        self.target_size = self.recognition_model.input_shape

        # detect faces from the template image
        template_img = cv2.imread(template_img_path)
        template_img = cv2.flip(template_img, 1)
        template_img = np.array(template_img)
        template_faces = DeepFace.extract_faces(template_img, target_size=self.target_size, detector_backend=FACE_DETECTOR_BACKEND, enforce_detection=False)
        # TODO: fix this to recalibrate if face is not detected
        if len(template_faces) == 0:
            sys.exit("No face detected in the template image")
        else:
            template = template_faces[0]["face"]
        template = np.expand_dims(template, axis=0)  # to (1, 224, 224, 3)
        template_embedding = self.recognition_model.find_embeddings(template)
        self.template_embedding = np.array(template_embedding)
        self.threshold = verification.find_threshold(model_name=FACE_RECOGNITION_MODEL_NAME, distance_metric=FACE_DISTANCE_METRIC)


    def recognize_face(self, face_img: np.ndarray) -> bool:
        if len(self.history) >= self.history_length:
            self.history.pop(0)
        self.history.append(False)

        faces = DeepFace.extract_faces(face_img, target_size=self.target_size, detector_backend=FACE_DETECTOR_BACKEND, enforce_detection=False)        
        for face in faces:
            face = np.expand_dims(face["face"], axis=0)
            face_embedding = self.recognition_model.find_embeddings(face)
            face_embedding = np.array(face_embedding)
            distance = verification.find_distance(self.template_embedding, face_embedding, distance_metric=FACE_DISTANCE_METRIC)

            if distance <= self.threshold:
                self.history[-1] = True
                break

        return any(self.history)