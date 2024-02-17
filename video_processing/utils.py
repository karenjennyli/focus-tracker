import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MOUTH_KEYPOINTS = [61, 39, 0, 269, 291, 405, 17, 181]
LEFT_EYE_KEYPOINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_KEYPOINTS = [362, 384, 387, 263, 373, 380]


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


def draw_landmarks(current_frame: np.ndarray, face_landmarks: list[vision.FaceLandmarker]) -> None:
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x,
                                        y=landmark.y,
                                        z=landmark.z) for
        landmark in
        face_landmarks
    ])

    # Create a new landmark list for the left eye keypoints
    left_eye_landmarks = landmark_pb2.NormalizedLandmarkList()
    left_eye_landmarks.landmark.extend([face_landmarks_proto.landmark[i] for i in LEFT_EYE_KEYPOINTS])

    # Draw the left eye keypoints
    mp_drawing.draw_landmarks(
        image=current_frame,
        landmark_list=left_eye_landmarks,
        connections=[],
        landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        connection_drawing_spec=None
    )

    # Create a new landmark list for the right eye keypoints
    right_eye_landmarks = landmark_pb2.NormalizedLandmarkList()
    right_eye_landmarks.landmark.extend([face_landmarks_proto.landmark[i] for i in RIGHT_EYE_KEYPOINTS])

    # Draw the right eye keypoints
    mp_drawing.draw_landmarks(
        image=current_frame,
        landmark_list=right_eye_landmarks,
        connections=[],
        landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        connection_drawing_spec=None
    )

    # Create a new landmark list for the mouth keypoints
    mouth_landmarks = landmark_pb2.NormalizedLandmarkList()
    mouth_landmarks.landmark.extend([face_landmarks_proto.landmark[i] for i in MOUTH_KEYPOINTS])

    # Draw the mouth keypoints
    mp_drawing.draw_landmarks(
        image=current_frame,
        landmark_list=mouth_landmarks,
        connections=[],
        landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        connection_drawing_spec=None
    )
