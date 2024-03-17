import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# drowsiness detection constants
CALIBRATION_TIME = 3
MOUTH_KEYPOINTS = [61, 39, 0, 269, 291, 405, 17, 181]
LEFT_EYE_KEYPOINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_KEYPOINTS = [362, 384, 387, 263, 373, 380]
DROWSINESS_HISTORY_LENGTH = 10
YAWN_MIN_TIME = 2
MICROSLEEP_MIN_TIME = 0.5

# gaze detection constants
HEAD_POSE_KEYPOINTS = [1, 9, 57, 130, 287, 359]
HEAD_POSE_3D_POINTS = np.array([
    [285, 528, 200],
    [285, 371, 152],
    [197, 574, 128],
    [173, 425, 108],
    [360, 574, 128],
    [391, 425, 108]
], dtype=np.float64)
GAZE_LEFT_THRESHOLD = -30
GAZE_RIGHT_THRESHOLD = 50
GAZE_HISTORY_LENGTH = 10
GAZE_MIN_TIME = 0.5

# phone detection constants
PHONE_MODEL_ID = 'phone-3ekgi/2'
PHONE_API_KEY = 'g2xhIxwutQ35mvQxltNJ'
PHONE_HISTORY_LENGTH = 10
PHONE_MIN_TIME = 1
PHONE_CONFIDENCE_THRESHOLD = 0.5

# all face landmark keypoints
ALL_KEYPOINTS = [MOUTH_KEYPOINTS, LEFT_EYE_KEYPOINTS, RIGHT_EYE_KEYPOINTS, HEAD_POSE_KEYPOINTS]

# result of the face landmark detection
FACE_DETECTION_RESULT = None

# calculate FPS
FPS_AVG_FRAME_COUNT = 10
COUNTER, FPS = 0, 0
START_TIME = time.time()


def euclidean_distance(p1: landmark_pb2.NormalizedLandmark, p2: landmark_pb2.NormalizedLandmark) -> float:
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


def mouth_aspect_ratio(face_landmarks: landmark_pb2.NormalizedLandmarkList) -> float:
    keypoints = MOUTH_KEYPOINTS
    p = [face_landmarks[keypoints[i]] for i in range(len(keypoints))]
    a = euclidean_distance(p[1], p[7])
    b = euclidean_distance(p[2], p[6])
    c = euclidean_distance(p[3], p[5])
    d = euclidean_distance(p[0], p[4])
    return (a + b + c) / (2.0 * d)


def eye_aspect_ratio(face_landmarks: landmark_pb2.NormalizedLandmarkList) -> float:
    keypoints = LEFT_EYE_KEYPOINTS
    p = [face_landmarks[keypoints[i]] for i in range(len(keypoints))]
    a = euclidean_distance(p[1], p[5])
    b = euclidean_distance(p[2], p[4])
    c = euclidean_distance(p[0], p[3])
    left_ear = (a + b) / (2.0 * c)
    
    keypoints = RIGHT_EYE_KEYPOINTS
    p = [face_landmarks[keypoints[i]] for i in range(len(keypoints))]
    a = euclidean_distance(p[1], p[5])
    b = euclidean_distance(p[2], p[4])
    c = euclidean_distance(p[0], p[3])
    right_ear = (a + b) / (2.0 * c)

    return (left_ear + right_ear) / 2.0


def draw_face_landmarks(current_frame: np.ndarray, face_landmarks: list[vision.FaceLandmarker]) -> None:
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x,
                                        y=landmark.y,
                                        z=landmark.z) for
        landmark in
        face_landmarks
    ])

    # Draw the face keypoints
    for keypoints in ALL_KEYPOINTS:
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend([face_landmarks_proto.landmark[i] for i in keypoints])
        mp_drawing.draw_landmarks(
            image=current_frame,
            landmark_list=landmark_list,
            connections=[],
            landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            connection_drawing_spec=None
        )
