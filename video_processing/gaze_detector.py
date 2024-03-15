# https://github.com/shenasa-ai/head-pose-estimation/tree/main

import time
import math
import numpy as np
import cv2

from utils import HEAD_POSE_KEYPOINTS, HEAD_POSE_3D_POINTS, GAZE_LEFT_THRESHOLD, GAZE_RIGHT_THRESHOLD, GAZE_HISTORY_LENGTH

class GazeDetector:

    def __init__(self, width: int, height: int, min_time: int, history_length: int = GAZE_HISTORY_LENGTH):
        self.width = width
        self.height = height
        self.min_time = min_time
        self.history_length = history_length
        self.yaw_history = []
        self.gazing_left = False
        self.gazing_right = False
        self.left_gaze_start_time = None
        self.right_gaze_start_time = None
        focal_length = width
        self.camera_matrix = np.array([[focal_length, 0, height // 2], [0, focal_length, width // 2], [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))


    def detect_gaze(self, face_landmarks: list) -> tuple[float, float, float]:
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
        self.yaw_history.append(yaw)
        if len(self.yaw_history) > self.history_length:
            self.yaw_history.pop(0)

        if np.mean(self.yaw_history) < GAZE_LEFT_THRESHOLD:
            if not self.gazing_left and self.left_gaze_start_time and time.time() - self.left_gaze_start_time >= self.min_time:
                self.gazing_left = True
                return 'left', pitch, yaw, roll
            elif not self.gazing_left and not self.left_gaze_start_time:
                self.left_gaze_start_time = time.time()
        else:
            self.gazing_left = False
            self.left_gaze_start_time = None

        if np.mean(self.yaw_history) > GAZE_RIGHT_THRESHOLD:
            if not self.gazing_right and self.right_gaze_start_time and time.time() - self.right_gaze_start_time >= self.min_time:
                self.gazing_right = True
                return 'right', pitch, yaw, roll
            elif not self.gazing_right and not self.right_gaze_start_time:
                self.right_gaze_start_time = time.time()
        else:
            self.gazing_right = False
            self.right_gaze_start_time = None

        return None, pitch, yaw, roll
