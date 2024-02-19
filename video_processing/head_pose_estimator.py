# https://github.com/shenasa-ai/head-pose-estimation/tree/main

import math
import numpy as np
import cv2

HEAD_POSE_KEYPOINTS = [1, 9, 57, 130, 287, 359]
HEAD_POSE_3D_POINTS = np.array([
    [285, 528, 200],
    [285, 371, 152],
    [197, 574, 128],
    [173, 425, 108],
    [360, 574, 128],
    [391, 425, 108]
], dtype=np.float64)

class HeadPoseEstimator:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        focal_length = width
        self.camera_matrix = np.array([[focal_length, 0, height // 2], [0, focal_length, width // 2], [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))

    def estimate_head_pose(self, face_landmarks: list) -> tuple[float, float, float]:
        # Get the 2D and 3D coordinates of the head pose keypoints
        points_2d = np.array([[int(landmark.x * self.width), int(landmark.y * self.height)] for landmark in [face_landmarks[i] for i in HEAD_POSE_KEYPOINTS]], dtype=np.float64)
        points_3d = HEAD_POSE_3D_POINTS

        # Solve the PnP problem
        success, rotation_vector, _ = cv2.solvePnP(points_3d, points_2d, self.camera_matrix, self.dist_coeffs)
        if not success:
            return None

        # Get rotational matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Get angles
        R = rotation_matrix
        pitch = math.atan2(R[2,1], R[2,2])
        yaw = math.atan2(-R[2,0], math.sqrt(R[2,1] ** 2 + R[2,2] ** 2))
        roll = math.atan2(R[1,0], R[0,0])
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)

        return pitch, yaw, roll
