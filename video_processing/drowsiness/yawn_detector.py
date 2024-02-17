import time
import numpy as np

from mediapipe.framework.formats import landmark_pb2

from utils import mouth_aspect_ratio

class YawnDetector:

    def __init__(self, min_time: int, mar_mean: float, mar_std: float, threshold: float, history_length: int):
        self.min_time = min_time
        self.mar_mean = mar_mean
        self.mar_std = mar_std
        self.threshold = threshold
        self.history_length = history_length
        self.history = []
        self.yawning = False
        self.yawn_start_time = None

    
    def detect_yawn(self, face_landmarks: landmark_pb2.NormalizedLandmarkList) -> tuple[bool, float]:
        # Calculate MAR, normalize, and add to history
        mar = mouth_aspect_ratio(face_landmarks)
        mar = (mar - self.mar_mean) / self.mar_std
        self.history.append(mar)
        if len(self.history) > self.history_length:
            self.history.pop(0)

        # Check if yawn
        if np.mean(self.history) > self.threshold:
            if not self.yawning and self.yawn_start_time and time.time() - self.yawn_start_time >= self.min_time:
                self.yawning = True
                return True, mar
            elif not self.yawning and not self.yawn_start_time:
                self.yawn_start_time = time.time()
        else:
            self.yawning = False
            self.yawn_start_time = None

        return False, mar