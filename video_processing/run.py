# https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/raspberry_pi/detect.py

import argparse
import sys
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import mouth_aspect_ratio, eye_aspect_ratio, draw_face_landmarks, draw_hand_landmarks
from utils import CALIBRATION_TIME, YAWN_MIN_TIME, MICROSLEEP_MIN_TIME, GAZE_MIN_TIME, PHONE_MIN_TIME
from utils import FPS_AVG_FRAME_COUNT, COUNTER, FPS, START_TIME
from utils import FACE_DETECTION_RESULT, HAND_DETECTION_RESULT

from people_detector import PeopleDetector
from yawn_detector import YawnDetector
from microsleep_detector import MicrosleepDetector
from gaze_detector import GazeDetector
from phone_detector import PhoneDetector

import requests
import uuid
import pytz

# Result of the face landmark detection
DETECTION_RESULT = None

# Calibration time
CALIBRATION_TIME = 3

# Calculate FPS
FPS_AVG_FRAME_COUNT = 10
COUNTER, FPS = 0, 0
START_TIME = time.time()

session_id = str(uuid.uuid4())


def capture_face_landmarks(cap: cv2.VideoCapture, face_landmarker: vision.FaceLandmarker,
                           calibration_time: int, width: int, height: int, calibration_message: str) -> list[tuple[float, float]]:
    start_time = None
    aspect_ratio_values = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        face_landmarker.detect_async(mp_image, time.time_ns() // 1_000_000)
        current_frame = image

        # Display the FPS on the image
        cv2.putText(current_frame, 'FPS: {:.2f}'.format(FPS), (10, height - 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if FACE_DETECTION_RESULT and FACE_DETECTION_RESULT.face_landmarks:
            draw_face_landmarks(current_frame, FACE_DETECTION_RESULT.face_landmarks[0])

        if start_time is None:
            cv2.putText(current_frame, calibration_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('video_processing', current_frame)
            if cv2.waitKey(1) == 32:
                start_time = time.time()
        else:
            if FACE_DETECTION_RESULT and FACE_DETECTION_RESULT.face_landmarks:
                ear = eye_aspect_ratio(FACE_DETECTION_RESULT.face_landmarks[0])
                mar = mouth_aspect_ratio(FACE_DETECTION_RESULT.face_landmarks[0])
                aspect_ratio_values.append((ear, mar))
                draw_face_landmarks(current_frame, FACE_DETECTION_RESULT.face_landmarks[0])

            cv2.putText(current_frame, 'Finished in: ' + str(calibration_time - int(time.time() - start_time)), (width // 2 - 100, height // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('video_processing', current_frame)
            cv2.waitKey(1)

            if time.time() - start_time >= calibration_time:
                break

    return aspect_ratio_values

def calibrate_face(cap: cv2.VideoCapture, face_landmarker: vision.FaceLandmarker, width: int, height: int) -> tuple[float, float, float, float]:
    neutral_face_message = f'Keep a neutral face with eyes open for {CALIBRATION_TIME} seconds. Press the space bar to start.'
    neutral_ear_values, neutral_mar_values = zip(*capture_face_landmarks(cap, face_landmarker, CALIBRATION_TIME, width, height, neutral_face_message))

    yawn_message = f'Yawn for {CALIBRATION_TIME} seconds. Press the space bar to start.'
    _, yawn_mar_values = zip(*capture_face_landmarks(cap, face_landmarker, CALIBRATION_TIME, width, height, yawn_message))

    eye_close_message = f'Close your eyes for {CALIBRATION_TIME} seconds. Press the space bar to start.'
    eye_close_ear_values, _ = zip(*capture_face_landmarks(cap, face_landmarker, CALIBRATION_TIME, width, height, eye_close_message))

    # Calculate the mean and standard deviation of the aspect ratios
    neutral_ear_mean, neutral_ear_std = np.mean(neutral_ear_values), np.std(neutral_ear_values)
    neutral_mar_mean, neutral_mar_std = np.mean(neutral_mar_values), np.std(neutral_mar_values)

    # Calculate the thresholds for the eye aspect ratio and mouth aspect ratio
    ear_threshold = np.mean(eye_close_ear_values) + np.mean(eye_close_ear_values) * 0.25
    ear_threshold = (ear_threshold - neutral_ear_mean) / neutral_ear_std
    mar_threshold = np.mean(yawn_mar_values) - np.mean(yawn_mar_values) * 0.1
    mar_threshold = (mar_threshold - neutral_mar_mean) / neutral_mar_std

    # Return the mean and standard deviation of the aspect ratios
    return neutral_ear_mean, neutral_ear_std, ear_threshold, neutral_mar_mean, neutral_mar_std, mar_threshold


def run(face_model: str, num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        hand_model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float,
        camera_id: int, width: int, height: int,
        drowsiness_enabled: bool, gaze_enabled: bool, phone_enabled: bool, hand_enabled: bool,
        django_enabled: bool) -> None:

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def save_face_result(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, FACE_DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % FPS_AVG_FRAME_COUNT == 0:
            FPS = FPS_AVG_FRAME_COUNT / (time.time() - START_TIME)
            START_TIME = time.time()

        FACE_DETECTION_RESULT = result
        COUNTER += 1

    # Initialize the face landmarker model
    face_base_options = python.BaseOptions(model_asset_path=face_model)
    options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        result_callback=save_face_result)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def save_hand_result(result: vision.HandLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global HAND_DETECTION_RESULT
        HAND_DETECTION_RESULT = result

    # Initialize the hand landmarker model
    base_options = python.BaseOptions(model_asset_path=hand_model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_hand_result)
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    # Calibrate the eye aspect ratio and mouth aspect ratio thresholds
    if drowsiness_enabled:
        ear_mean, ear_std, ear_threshold, mar_mean, mar_std, mar_threshold = calibrate_face(cap, face_landmarker, width, height)
        print(f'EAR threshold: {ear_threshold}, MAR threshold: {mar_threshold}')

    # Initialize detectors
    people_detector = PeopleDetector(min_time=PHONE_MIN_TIME)
    if drowsiness_enabled:
        yawn_detector = YawnDetector(min_time=YAWN_MIN_TIME, mar_mean=mar_mean, mar_std=mar_std, threshold=mar_threshold)
        microsleep_detector = MicrosleepDetector(min_time=MICROSLEEP_MIN_TIME, ear_mean=ear_mean, ear_std=ear_std, threshold=ear_threshold)
    if gaze_enabled:
        gaze_detector = GazeDetector(width=width, height=height, min_time=GAZE_MIN_TIME)
    if phone_enabled:
        phone_detector = PhoneDetector(width=width, height=height, min_time=PHONE_MIN_TIME)

    # Wait for the user to press the space bar to start the program
    while True:
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)
        current_frame = image
        cv2.putText(current_frame, 'Press the space bar to start the program.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('video_processing', image)
        if cv2.waitKey(1) == 32:
            break

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)
        current_frame = image

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Run the face landmark detection model
        face_landmarker.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Run the hand landmark detection model
        if hand_enabled:
            hand_landmarker.detect_async(mp_image, time.time_ns() // 1_000_000)
        
        # Display the FPS on the image
        cv2.putText(current_frame, 'FPS: {:.2f}'.format(FPS), (10, height - 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # TODO: implement facial recognition to distinguish between user's face and other faces

        if FACE_DETECTION_RESULT and FACE_DETECTION_RESULT.face_landmarks:
            if django_enabled:
                current_session_data = {
                    'session_id': session_id,
                }
                resp = requests.post('http://127.0.0.1:8000/api/current_session', json=current_session_data)
                # if resp.status_code == 201:
                #         print("Current_session data successfully sent to Django")

            people_detected = people_detector.detect_people(FACE_DETECTION_RESULT.face_landmarks)
            if people_detected:
                print(f'Other people detected: ', datetime.now().strftime('%H:%M:%S'))

            face_landmarks = FACE_DETECTION_RESULT.face_landmarks[0]

            if drowsiness_enabled:
                yawn_detected, mar = yawn_detector.detect_yawn(face_landmarks)
                if yawn_detected:
                    print(f'Yawn: ', datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'MAR: ', mar)
                    if django_enabled:
                        now_utc = datetime.now(pytz.utc)
                        now_eastern = now_utc.astimezone(pytz.timezone('America/New_York'))
                        data = {
                            'session_id': session_id,
                            'user_id': 'user123',
                            'detection_type': 'yawn',
                            'timestamp': now_eastern.strftime('%Y-%m-%dT%H:%M:%S'),
                            'aspect_ratio': mar,  # Mouth Aspect Ratio for yawn detection
                        }
                        response = requests.post('http://127.0.0.1:8000/api/detections/', json=data)
                        if response.status_code == 201:
                            print("Yawn data successfully sent to Django")

                microsleep_detected, ear = microsleep_detector.detect_microsleep(face_landmarks)
                if microsleep_detected:
                    print(f'Microsleep: ', datetime.now().strftime('%H:%M:%S'), 'EAR: ', ear)

            if gaze_enabled:
                gaze, pitch, yaw, roll = gaze_detector.detect_gaze(face_landmarks)
                if gaze == 'left' or gaze == 'right':
                    print(f'Gaze: ', datetime.now().strftime('%H:%M:%S'), gaze)

            # Display the aspect ratios on the image
            if drowsiness_enabled:
                cv2.putText(current_frame, 'Eye aspect ratio: {:.2f}'.format(ear), 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(current_frame, 'Mouth aspect ratio: {:.2f}'.format(mar), 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw the head pose angles at the top left corner of the image
            if gaze_enabled:
                cv2.putText(current_frame, f'Yaw: {int(yaw)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            draw_face_landmarks(current_frame, face_landmarks)

        if HAND_DETECTION_RESULT and HAND_DETECTION_RESULT.hand_landmarks:
            hand_landmarks = HAND_DETECTION_RESULT.hand_landmarks
            draw_hand_landmarks(current_frame, hand_landmarks)
        else:
            hand_landmarks = None

        if phone_enabled:
            phone_detected, annotated_image = phone_detector.detect_phone(current_frame, hand_landmarks)
            if phone_detected:
                print(f'Phone: ', datetime.now().strftime('%H:%M:%S'))
                current_frame = annotated_image

        
        cv2.imshow('video_processing', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    face_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--face_model',
        help='Name of face landmarker model.',
        required=False,
        default='face_landmarker.task')
    parser.add_argument(
        '--numFaces',
        help='Max number of faces that can be detected by the landmarker.',
        required=False,
        default=2)
    parser.add_argument(
        '--minFaceDetectionConfidence',
        help='The minimum confidence score for face detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minFacePresenceConfidence',
        help='The minimum confidence score of face presence score in the face '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the face tracking to be '
             'considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--hand_model',
        help='Name of the hand landmarker model bundle.',
        required=False,
        default='hand_landmarker.task')
    parser.add_argument(
        '--numHands',
        help='Max number of hands that can be detected by the landmarker.',
        required=False,
        default=2)
    parser.add_argument(
        '--minHandDetectionConfidence',
        help='The minimum confidence score for hand detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minHandPresenceConfidence',
        help='The minimum confidence score of hand presence score in the hand '
             'landmark detection.',
        required=False,
        default=0.5)
    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=1920)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=1080)
    parser.add_argument(
        '--disableDrowsiness',
        help='Enable drowsiness detection.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--disableGaze',
        help='Enable gaze detection.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--disablePhone',
        help='Enable phone detection.',
        action='store_true',
        required=False,
        default=False
    )
    parser.add_argument(
        '--disableHand',
        help='Enable hand detection.',
        action='store_true',
        required=False,
        default=False
    )
    parser.add_argument(
        '--disableDjango',
        help='Enable Django server.',
        action='store_true',
        required=False,
        default=False
    )
    args = parser.parse_args()

    run(args.face_model, int(args.numFaces), args.minFaceDetectionConfidence,
        args.minFacePresenceConfidence, args.minTrackingConfidence,
        args.hand_model, int(args.numHands), args.minHandDetectionConfidence,
        args.minHandPresenceConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight,
        not args.disableDrowsiness, not args.disableGaze, not args.disablePhone, not args.disableHand,
        not args.disableDjango)


if __name__ == '__main__':
    main()