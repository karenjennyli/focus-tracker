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

    def __init__(self, width: int, height: int, time_limit: int = 30, template_img_path: str = 'calibration_data/template_face.jpg'):
        self.img_width = width
        self.img_height = height
        self.time_limit = time_limit
        self.last_match_time = None
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
        # returns True if the face is recognized, False otherwise
        recognized = False
        faces = DeepFace.extract_faces(face_img, target_size=self.target_size, detector_backend=FACE_DETECTOR_BACKEND, enforce_detection=False)
        if len(faces) == 0:
            return recognized

        for face in faces:
            face = np.expand_dims(face["face"], axis=0)
            face_embedding = self.recognition_model.find_embeddings(face)
            face_embedding = np.array(face_embedding)
            distance = verification.find_distance(self.template_embedding, face_embedding, distance_metric=FACE_DISTANCE_METRIC)

            if distance <= self.threshold:
                recognized = True
                break

        return recognized
    

    def get_face_coords(self, face_landmarks: landmark_pb2.NormalizedLandmarkList) -> tuple[int, int, int]:
        top_landmark = face_landmarks[10]
        bottom_landmark = face_landmarks[152]
        left_landmark = face_landmarks[127]
        right_landmark = face_landmarks[356]
        top = int(top_landmark.y * self.img_height)
        bottom = int(bottom_landmark.y * self.img_height)
        left = int(left_landmark.x * self.img_width)
        right = int(right_landmark.x * self.img_width)
        x = int((left + right) / 2)
        y = int((top + bottom) / 2)
        size = int(((right - left) ** 2 + (bottom - top) ** 2) ** 0.5)
        return x, y, size

    
    def get_face_index(self, face_img: np.ndarray, face_landmarks: list[landmark_pb2.NormalizedLandmarkList]) -> int:
        # returns index of the matched face, -1 if no match
        recognized_index = -1
        faces = DeepFace.extract_faces(face_img, target_size=self.target_size, detector_backend=FACE_DETECTOR_BACKEND, enforce_detection=False)
        if len(faces) == 0:
            return recognized_index
        
        for i, face in enumerate(faces):
            face = np.expand_dims(face["face"], axis=0)
            face_embedding = self.recognition_model.find_embeddings(face)
            face_embedding = np.array(face_embedding)
            distance = verification.find_distance(self.template_embedding, face_embedding, distance_metric=FACE_DISTANCE_METRIC)

            if distance <= self.threshold:
                recognized_index = i
                size = face['facial_area']['w'] * face['facial_area']['h']
                x = face['facial_area']['x'] + face['facial_area']['w'] // 2
                y = face['facial_area']['y'] + face['facial_area']['h'] // 2
                deepface_coords = (x, y, size)
                break

        for this_face_landmarks in face_landmarks:
            x, y, size = self.get_face_coords(this_face_landmarks)
            face_coords = (x, y, size)
            if np.linalg.norm(np.array(deepface_coords[:2]) - np.array(face_coords[:2])) <= 50:
                return recognized_index
