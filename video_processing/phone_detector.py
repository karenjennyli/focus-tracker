# https://supervision.roboflow.com/latest/how_to/detect_and_annotate/

from inference import get_roboflow_model
import supervision as sv
import time
import numpy as np

from utils import PHONE_MODEL_ID, PHONE_API_KEY, PHONE_CONFIDENCE_THRESHOLD


class PhoneDetector:

    def __init__(self, min_time: int, history_length: int = 5, confidence: float = PHONE_CONFIDENCE_THRESHOLD):
        self.min_time = min_time
        self.history_length = history_length
        self.history = []
        self.phone_detected = False
        self.phone_start_time = None
        self.model = get_roboflow_model(model_id=PHONE_MODEL_ID, api_key=PHONE_API_KEY)
        self.confidence = confidence
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.percentage_bar_annotator = sv.PercentageBarAnnotator()

    
    def detect_phone(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        results = self.model.infer(frame, confidence=self.confidence)
        detections = sv.Detections.from_inference(
            results[0].dict(by_alias=True, exclude_none=True)
        )
        annotated_image = self.bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = self.percentage_bar_annotator.annotate(
            scene=annotated_image, detections=detections
        )

        self.history.append(detections)
        if len(self.history) > self.history_length:
            self.history.pop(0)
            
        if self.holding_phone(self.history):
            if not self.phone_detected and self.phone_start_time and time.time() - self.phone_start_time >= self.min_time:
                self.phone_detected = True
                return True, annotated_image
            elif not self.phone_detected and not self.phone_start_time:
                self.phone_start_time = time.time()
        else:
            self.phone_detected = False
            self.phone_start_time = None

        return False, annotated_image


    def holding_phone(self, detections_history: list[sv.Detections]) -> bool:
        '''
        Count number of phone detections in history
        Example Detections:
        `Detections(xyxy=array([], shape=(0, 4), dtype=float32), mask=None, confidence=array([], dtype=float32), class_id=array([], dtype=int64), tracker_id=None, data={})`
        `Detections(xyxy=array([[       1201,         417,        1588,         814]]), mask=None, confidence=array([     0.7198]), class_id=array([0]), tracker_id=None, data={'class_name': array(['phone'], dtype='<U5')})`

        keep count of how many actual phone detections are in the history

        for each detections in detections_history:
            for each detection in detections:
                if location of phone is close to location of hand:
                    increment count

        if the count is greater than 80% of the history length, return True

        '''
        count = 0
        for detections in detections_history:
            if len(detections) > 0:
                count += 1
        
        return count >= self.history_length * 0.8