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
from mediapipe.framework.formats import landmark_pb2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DETECTION_RESULT = None

# Keypoints from the face mesh model:
# https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png

# Order: left, upper left, upper right, right, lower right, lower left
LEFT_EYE_KEYPOINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_KEYPOINTS = [362, 384, 387, 263, 373, 380]

# Order: left, upper left, top, upper right, right, lower right, bottom, lower left
MOUTH_KEYPOINTS = [61, 39, 0, 269, 291, 405, 17, 181]

# Thresholds for the eye aspect ratio and mouth aspect ratio
EAR_THRESHOLD = None
MAR_THRESHOLD = None

# Aspect ratios from previous frames
ear_history = []
mar_history = []

# The maximum length of the aspect ratio history
HISTORY_LENGTH = 10

# Length of calibration
CALIBRATION_TIME = 3

# Minimum time for yawn and eye close detection in seconds
YAWN_MIN_TIME = 3
MICROSLEEP_MIN_TIME = 1

# Calculate FPS
FPS_AVG_FRAME_COUNT = 10
COUNTER, FPS = 0, 0
START_TIME = time.time()


def euclidean_distance(p1: landmark_pb2.NormalizedLandmark, p2: landmark_pb2.NormalizedLandmark) -> float:
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


def eye_aspect_ratio(face_landmarks: landmark_pb2.NormalizedLandmarkList, keypoints: list[int]) -> float:
    p = [face_landmarks[keypoints[i]] for i in range(len(keypoints))]
    a = euclidean_distance(p[1], p[5])
    b = euclidean_distance(p[2], p[4])
    c = euclidean_distance(p[0], p[3])
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(face_landmarks: landmark_pb2.NormalizedLandmarkList, keypoints: list[int]) -> float:
    p = [face_landmarks[keypoints[i]] for i in range(len(keypoints))]
    a = euclidean_distance(p[1], p[7])
    b = euclidean_distance(p[2], p[6])
    c = euclidean_distance(p[3], p[5])
    d = euclidean_distance(p[0], p[4])
    return (a + b + c) / (2.0 * d)


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
        cv2.putText(current_frame, "FPS: {:.2f}".format(FPS), (10, height - 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
            draw_landmarks(current_frame, DETECTION_RESULT.face_landmarks[0])

        if start_time is None:
            cv2.putText(current_frame, calibration_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('face_landmarker', current_frame)
            if cv2.waitKey(1) == 32:
                start_time = time.time()
        else:
            if DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
                left_ear = eye_aspect_ratio(DETECTION_RESULT.face_landmarks[0], LEFT_EYE_KEYPOINTS)
                right_ear = eye_aspect_ratio(DETECTION_RESULT.face_landmarks[0], RIGHT_EYE_KEYPOINTS)
                mar = mouth_aspect_ratio(DETECTION_RESULT.face_landmarks[0], MOUTH_KEYPOINTS)
                aspect_ratio_values.append(((left_ear + right_ear) / 2, mar))
                draw_landmarks(current_frame, DETECTION_RESULT.face_landmarks[0])

            cv2.putText(current_frame, "Finished in: " + str(calibration_time - int(time.time() - start_time)), (width // 2 - 100, height // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('face_landmarker', current_frame)
            cv2.waitKey(1)

            if time.time() - start_time >= calibration_time:
                break

    return aspect_ratio_values

def calibrate(cap: cv2.VideoCapture, detector: vision.FaceLandmarker, width: int, height: int) -> tuple[float, float, float, float]:
    neutral_face_message = f"Keep a neutral face with eyes open for {CALIBRATION_TIME} seconds. Press the space bar to start."
    neutral_ear_values, neutral_mar_values = zip(*capture_face_landmarks(cap, detector, CALIBRATION_TIME, width, height, neutral_face_message))

    yawn_message = f"Yawn for {CALIBRATION_TIME} seconds. Press the space bar to start."
    _, yawn_mar_values = zip(*capture_face_landmarks(cap, detector, CALIBRATION_TIME, width, height, yawn_message))

    eye_close_message = f"Close your eyes for {CALIBRATION_TIME} seconds. Press the space bar to start."
    eye_close_ear_values, _ = zip(*capture_face_landmarks(cap, detector, CALIBRATION_TIME, width, height, eye_close_message))

    # Calculate the mean and standard deviation of the aspect ratios
    neutral_ear_mean, neutral_ear_std = np.mean(neutral_ear_values), np.std(neutral_ear_values)
    neutral_mar_mean, neutral_mar_std = np.mean(neutral_mar_values), np.std(neutral_mar_values)

    # Calculate the thresholds for the eye aspect ratio and mouth aspect ratio
    global EAR_THRESHOLD, MAR_THRESHOLD
    EAR_THRESHOLD = np.mean(eye_close_ear_values) + np.mean(eye_close_ear_values) * 0.1
    EAR_THRESHOLD = (EAR_THRESHOLD - neutral_ear_mean) / neutral_ear_std
    MAR_THRESHOLD = np.mean(yawn_mar_values) - np.mean(yawn_mar_values) * 0.1
    MAR_THRESHOLD = (MAR_THRESHOLD - neutral_mar_mean) / neutral_mar_std

    # Return the mean and standard deviation of the aspect ratios
    return neutral_ear_mean, neutral_ear_std, neutral_mar_mean, neutral_mar_std


def run(model: str, num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

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
  """

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
    ear_mean, ear_std, mar_mean, mar_std = calibrate(cap, detector, width, height)

    # Wait for the user to press the space bar to start the program
    while True:
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)
        current_frame = image
        cv2.putText(current_frame, "Press the space bar to start the program.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('face_landmarker', image)
        if cv2.waitKey(1) == 32:
            break

    # Initialize start times for microsleep and yawning detection
    microsleep_start_time = None
    yawn_start_time = None

    # Add a flag for eye closure
    eyes_closed = False

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run face landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        current_frame = image
        
        # Display the FPS on the image
        cv2.putText(current_frame, "FPS: {:.2f}".format(FPS), (10, height - 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
            face_landmarks = DETECTION_RESULT.face_landmarks[0]

            # Calculate the aspect ratios for the left and right eyes
            left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE_KEYPOINTS)
            right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE_KEYPOINTS)
            mar = mouth_aspect_ratio(face_landmarks, MOUTH_KEYPOINTS)

            # Normalize the aspect ratios
            left_ear = (left_ear - ear_mean) / ear_std
            right_ear = (right_ear - ear_mean) / ear_std
            mar = (mar - mar_mean) / mar_std

            # Append the aspect ratios to the history
            ear_history.append(left_ear)
            mar_history.append(mar)

            # Remove the oldest aspect ratios if the history is too long
            if len(ear_history) > HISTORY_LENGTH:
                ear_history.pop(0)
            if len(mar_history) > HISTORY_LENGTH:
                mar_history.pop(0)

            # Check if microsleep is detected
            if np.mean(ear_history) < EAR_THRESHOLD:
                if not eyes_closed and microsleep_start_time and time.time() - microsleep_start_time >= MICROSLEEP_MIN_TIME:
                    eyes_closed = True
                    print(f"Microsleep: ", datetime.now().strftime("%H:%M:%S"), "EAR: ", np.mean(ear_history))
                elif not eyes_closed and not microsleep_start_time:
                    microsleep_start_time = time.time()
            else:
                eyes_closed = False
                microsleep_start_time = None

            # Check if yawn is detected
            if np.mean(mar_history) > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time >= YAWN_MIN_TIME:
                    print(f"Yawn: ", datetime.now().strftime("%H:%M:%S"), "MAR: ", np.mean(mar_history))
                    yawn_start_time = None
            else:
                yawn_start_time = None

            # Display the aspect ratios on the image
            cv2.putText(current_frame, "Left eye aspect ratio: {:.2f}".format(left_ear), 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(current_frame, "Right eye aspect ratio: {:.2f}".format(right_ear), 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(current_frame, "Mouth aspect ratio: {:.2f}".format(mar), 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the calibration thresholds on the image
            cv2.putText(current_frame, "EAR_THRESHOLD: {:.2f}".format(EAR_THRESHOLD),
                        (width - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(current_frame, "MAR_THRESHOLD: {:.2f}".format(MAR_THRESHOLD),
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