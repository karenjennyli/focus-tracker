# https://supervision.roboflow.com/latest/how_to/detect_and_annotate/

import supervision as sv
import time
import numpy as np

from inference import get_roboflow_model
from mediapipe.framework.formats import landmark_pb2

from utils import PHONE_MODEL_ID, PHONE_API_KEY, PHONE_CONFIDENCE_THRESHOLD


class PhoneDetector:

    def __init__(self, width: int, height: int, min_time: int, history_length: int = 5,
                 confidence: float = PHONE_CONFIDENCE_THRESHOLD):
        self.width = width
        self.height = height
        self.min_time = min_time
        self.history_length = history_length
        self.phones_history = []
        self.hands_history = []
        self.phone_detected = False
        self.phone_start_time = None
        self.model = get_roboflow_model(model_id=PHONE_MODEL_ID, api_key=PHONE_API_KEY)
        self.confidence = confidence
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.percentage_bar_annotator = sv.PercentageBarAnnotator()


    def get_phones_coords(self, detections: sv.Detections) -> list[tuple[int, int]]:
        phones_coords = []
        for x1, y1, x2, y2 in detections.xyxy:
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            size = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            phones_coords.append((x, y, size))
        return phones_coords
    

    def get_hands_coords(self, hand_landmarks: list[landmark_pb2.NormalizedLandmarkList]) -> list[tuple[int, int]]:
        hands_coords = []
        for landmarks in hand_landmarks:
            x = int(sum([landmark.x * self.width for landmark in landmarks]) / len(landmarks))
            y = int(sum([landmark.y * self.height for landmark in landmarks]) / len(landmarks))
            hands_coords.append((x, y))
        return hands_coords

    
    def detect_phone(self, frame: np.ndarray, hands_landmarks: list[landmark_pb2.NormalizedLandmarkList]) -> tuple[bool, np.ndarray]:
        results = self.model.infer(frame, confidence=self.confidence)
        detections = sv.Detections.from_inference(
            results[0].dict(by_alias=True, exclude_none=True)
        )
        annotated_image = self.bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = self.percentage_bar_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        phones_coords = self.get_phones_coords(detections)
        self.phones_history.append(phones_coords)
        if len(self.phones_history) > self.history_length:
            self.phones_history.pop(0)
        
        if hands_landmarks:
            hands_coords = self.get_hands_coords(hands_landmarks)
            self.hands_history.append(hands_coords)
            if len(self.hands_history) > self.history_length:
                self.hands_history.pop(0)
        else:
            self.hands_history.append([(-1, -1)])
            if len(self.hands_history) > self.history_length:
                self.hands_history.pop(0)
            
        if self.holding_phone(self.phones_history, self.hands_history):
            # TODO: do some check so that we don't duplicate detections within x (5?) seconds of each other
            if not self.phone_detected and self.phone_start_time and time.time() - self.phone_start_time >= self.min_time:
                self.phone_detected = True
                return True, annotated_image
            elif not self.phone_detected and not self.phone_start_time:
                self.phone_start_time = time.time()
        else:
            self.phone_detected = False
            self.phone_start_time = None

        return False, annotated_image


    def holding_phone(self, phones_history: list[tuple[int, int]],
                      hands_history: list[landmark_pb2.NormalizedLandmarkList]) -> bool:
        count_valid = 0
        for phones_coords, hands_coords in zip(phones_history, hands_history):
            # check all combinations of phones and hands that their distance is less than threshold based on size of phone
            for phone in phones_coords:
                for hand in hands_coords:
                    distance = int(np.linalg.norm(np.array(phone[:2]) - np.array(hand)))
                    distance_threshold = int(phone[2] / 2)
                    print(distance, distance_threshold)
                    if hand != (-1, -1) and distance < distance_threshold:
                        count_valid += 1
                        break

        return count_valid >= self.history_length * 0.8
