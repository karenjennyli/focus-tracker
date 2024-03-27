import time

from mediapipe.framework.formats import landmark_pb2

from utils import PEOPLE_HISTORY_LENGTH

class PeopleDetector:

    def __init__(self, min_time: int, history_length: int = PEOPLE_HISTORY_LENGTH):
        self.min_time = min_time
        self.history_length = history_length
        self.history = []
        self.people_detected = False
        self.people_start_time = None

    
    def detect_people(self, face_landmarks: list[landmark_pb2.NormalizedLandmarkList]) -> bool:
        people_count = len(face_landmarks)
        self.history.append(people_count)
        if len(self.history) > self.history_length:
            self.history.pop(0)

        if all([count > 1 for count in self.history]):
            if not self.people_detected and self.people_start_time and time.time() - self.people_start_time >= self.min_time:
                self.people_detected = True
                return True
            elif not self.people_detected and not self.people_start_time:
                self.people_start_time = time.time()
        else:
            self.people_detected = False
            self.people_start_time = None

        return False
