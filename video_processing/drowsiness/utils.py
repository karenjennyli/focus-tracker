from mediapipe.framework.formats import landmark_pb2

MOUTH_KEYPOINTS = [61, 39, 0, 269, 291, 405, 17, 181]

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
