import time
import numpy as np

from mediapipe.framework.formats import landmark_pb2

from utils import eye_aspect_ratio

class MicrosleepDetector:

    def __init__(self, min_time: int, ear_mean: float, ear_std: float, threshold: float, history_length: int):
        self.min_time = min_time
        self.ear_mean = ear_mean
        self.ear_std = ear_std
        self.threshold = threshold
        self.history_length = history_length
        self.history = []
        self.eyes_closed = False
        self.microsleep_start_time = None

    def detect_microsleep(self, face_landmarks: landmark_pb2.NormalizedLandmarkList) -> tuple[bool, float]:
        # Calculate EAR, normalize, and add to history
        ear = eye_aspect_ratio(face_landmarks)
        ear = (ear - self.ear_mean) / self.ear_std
        self.history.append(ear)
        if len(self.history) > self.history_length:
            self.history.pop(0)

        # Check if microsleep
        if np.mean(self.history) < self.threshold:
            if not self.eyes_closed and self.microsleep_start_time and time.time() - self.microsleep_start_time >= self.min_time:
                self.eyes_closed = True
                return True, ear
            elif not self.eyes_closed and not self.microsleep_start_time:
                self.microsleep_start_time = time.time()
        else:
            self.eyes_closed = False
            self.microsleep_start_time = None

        return False, ear
