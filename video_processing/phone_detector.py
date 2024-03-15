# https://supervision.roboflow.com/latest/how_to/detect_and_annotate/

from inference import get_roboflow_model
import supervision as sv
import numpy as np

from utils import PHONE_MODEL_ID, PHONE_API_KEY, PHONE_CONFIDENCE_THRESHOLD


class PhoneDetector:

    def __init__(self, min_time: int, history_length: int = 10, confidence: float = PHONE_CONFIDENCE_THRESHOLD):
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
        return False, annotated_image
