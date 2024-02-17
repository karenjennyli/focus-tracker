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

from utils import mouth_aspect_ratio, eye_aspect_ratio, draw_landmarks
from yawn_detector import YawnDetector
from microsleep_detector import MicrosleepDetector

# Result of the face landmark detection
DETECTION_RESULT = None

# Calibration time
CALIBRATION_TIME = 3

# Calculate FPS
FPS_AVG_FRAME_COUNT = 10
COUNTER, FPS = 0, 0
START_TIME = time.time()


def capture_face_landmarks(cap, detector, calibration_time, width, height, calibration_message):
    start_time = None
    aspect_ratio_values = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        current_frame = image

        # Display the FPS on the image
        cv2.putText(current_frame, 'FPS: {:.2f}'.format(FPS), (10, height - 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
            draw_landmarks(current_frame, DETECTION_RESULT.face_landmarks[0])

        if start_time is None:
            cv2.putText(current_frame, calibration_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('face_landmarker', current_frame)
            if cv2.waitKey(1) == 32:
                start_time = time.time()
        else:
            if DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
                ear = eye_aspect_ratio(DETECTION_RESULT.face_landmarks[0])
                mar = mouth_aspect_ratio(DETECTION_RESULT.face_landmarks[0])
                aspect_ratio_values.append((ear, mar))
                draw_landmarks(current_frame, DETECTION_RESULT.face_landmarks[0])

            cv2.putText(current_frame, 'Finished in: ' + str(calibration_time - int(time.time() - start_time)), (width // 2 - 100, height // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('face_landmarker', current_frame)
            cv2.waitKey(1)

            if time.time() - start_time >= calibration_time:
                break

    return aspect_ratio_values

def calibrate(cap: cv2.VideoCapture, detector: vision.FaceLandmarker, width: int, height: int) -> tuple[float, float, float, float]:
    neutral_face_message = f'Keep a neutral face with eyes open for {CALIBRATION_TIME} seconds. Press the space bar to start.'
    neutral_ear_values, neutral_mar_values = zip(*capture_face_landmarks(cap, detector, CALIBRATION_TIME, width, height, neutral_face_message))

    yawn_message = f'Yawn for {CALIBRATION_TIME} seconds. Press the space bar to start.'
    _, yawn_mar_values = zip(*capture_face_landmarks(cap, detector, CALIBRATION_TIME, width, height, yawn_message))

    eye_close_message = f'Close your eyes for {CALIBRATION_TIME} seconds. Press the space bar to start.'
    eye_close_ear_values, _ = zip(*capture_face_landmarks(cap, detector, CALIBRATION_TIME, width, height, eye_close_message))

    # Calculate the mean and standard deviation of the aspect ratios
    neutral_ear_mean, neutral_ear_std = np.mean(neutral_ear_values), np.std(neutral_ear_values)
    neutral_mar_mean, neutral_mar_std = np.mean(neutral_mar_values), np.std(neutral_mar_values)

    # Calculate the thresholds for the eye aspect ratio and mouth aspect ratio
    ear_threshold = np.mean(eye_close_ear_values) + np.mean(eye_close_ear_values) * 0.1
    ear_threshold = (ear_threshold - neutral_ear_mean) / neutral_ear_std
    mar_threshold = np.mean(yawn_mar_values) - np.mean(yawn_mar_values) * 0.1
    mar_threshold = (mar_threshold - neutral_mar_mean) / neutral_mar_std

    # Return the mean and standard deviation of the aspect ratios
    return neutral_ear_mean, neutral_ear_std, ear_threshold, neutral_mar_mean, neutral_mar_std, mar_threshold


def run(model: str, num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    '''Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the face landmarker model bundle.
      num_faces: Max number of faces that can be detected by the landmarker.
      min_face_detection_confidence: The minimum confidence score for face
        detection to be considered successful.
      min_face_presence_confidence: The minimum confidence score of face
        presence score in the face landmark detection.
      min_tracking_confidence: The minimum confidence score for the face
        tracking to be considered successful.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  '''

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def save_result(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % FPS_AVG_FRAME_COUNT == 0:
            FPS = FPS_AVG_FRAME_COUNT / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    # Initialize the face landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=True,
        result_callback=save_result)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Calibrate the eye aspect ratio and mouth aspect ratio thresholds
    ear_mean, ear_std, ear_threshold, mar_mean, mar_std, mar_threshold = calibrate(cap, detector, width, height)

    # Create a yawn detector
    yawn_detector = YawnDetector(min_time=3, mar_mean=mar_mean, mar_std=mar_std, threshold=mar_threshold, history_length=10)

    # Create a microsleep detector
    microsleep_detector = MicrosleepDetector(min_time=1, ear_mean=ear_mean, ear_std=ear_std, threshold=ear_threshold, history_length=10)

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
        cv2.imshow('face_landmarker', image)
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

        # Run the face landmark detection model
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        
        # Display the FPS on the image
        cv2.putText(current_frame, 'FPS: {:.2f}'.format(FPS), (10, height - 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
            face_landmarks = DETECTION_RESULT.face_landmarks[0]

            yawn_detected, mar = yawn_detector.detect_yawn(face_landmarks)
            if yawn_detected:
                print(f'Yawn: ', datetime.now().strftime('%H:%M:%S'), 'MAR: ', mar)

            microsleep_detected, ear = microsleep_detector.detect_microsleep(face_landmarks)
            if microsleep_detected:
                print(f'Microsleep: ', datetime.now().strftime('%H:%M:%S'), 'EAR: ', ear)

            # Display the aspect ratios on the image
            cv2.putText(current_frame, 'Left eye aspect ratio: {:.2f}'.format(ear), 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(current_frame, 'Mouth aspect ratio: {:.2f}'.format(mar), 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the calibration thresholds on the image
            cv2.putText(current_frame, 'ear_threshold: {:.2f}'.format(ear_threshold),
                        (width - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(current_frame, 'mar_threshold: {:.2f}'.format(mar_threshold),
                        (width - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            draw_landmarks(current_frame, face_landmarks)

        cv2.imshow('face_landmarker', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of face landmarker model.',
        required=False,
        default='face_landmarker.task')
    parser.add_argument(
        '--numFaces',
        help='Max number of faces that can be detected by the landmarker.',
        required=False,
        default=1)
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
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=960)
    args = parser.parse_args()

    run(args.model, int(args.numFaces), args.minFaceDetectionConfidence,
        args.minFacePresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()