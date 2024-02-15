# https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/raspberry_pi/detect.py

import argparse
import sys
import time

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
left_eye_keypoints = [33, 160, 158, 133, 153, 144]
right_eye_keypoints = [362, 384, 387, 263, 373, 380]

# Order: left, upper left, top, upper right, right, lower right, bottom, lower left
mouth_keypoints = [61, 39, 0, 269, 291, 405, 17, 181]

# Thresholds for the eye aspect ratio and mouth aspect ratio
EAR_THRESHOLD = -5
MAR_THRESHOLD = 115

# Aspect ratios from previous frames
ear_history = []
mar_history = []

# The maximum length of the aspect ratio history
HISTORY_LENGTH = 30

# Number of frames to use for calibration
CALIBRATION_FRAME_COUNT = 200


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
    left_eye_landmarks.landmark.extend([face_landmarks_proto.landmark[i] for i in left_eye_keypoints])

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
    right_eye_landmarks.landmark.extend([face_landmarks_proto.landmark[i] for i in right_eye_keypoints])

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
    mouth_landmarks.landmark.extend([face_landmarks_proto.landmark[i] for i in mouth_keypoints])

    # Draw the mouth keypoints
    mp_drawing.draw_landmarks(
        image=current_frame,
        landmark_list=mouth_landmarks,
        connections=[],
        landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        connection_drawing_spec=None
    )


def calibrate(calibration_frame_count: int, cap: cv2.VideoCapture,
              detector: vision.FaceLandmarker) -> tuple[float, float, float, float]:
    """Calibrate the eye aspect ratio and mouth aspect ratio thresholds.

    Args:
        calibration_frame_count: The number of frames to use for calibration.
        detector: The face landmarker model.
    """

    # Initialize the calibration variables
    ear_values = []
    mar_values = []

    # Capture the calibration frames
    for i in range(calibration_frame_count):
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

        if DETECTION_RESULT:
            # Calculate the aspect ratios for the left and right eyes
            left_ear = eye_aspect_ratio(DETECTION_RESULT.face_landmarks[0], left_eye_keypoints)
            right_ear = eye_aspect_ratio(DETECTION_RESULT.face_landmarks[0], right_eye_keypoints)
            mar = mouth_aspect_ratio(DETECTION_RESULT.face_landmarks[0], mouth_keypoints)

            # Update the calibration variables
            ear_values.append((left_ear + right_ear) / 2)
            mar_values.append(mar)


            draw_landmarks(current_frame, DETECTION_RESULT.face_landmarks[0])
        
        # Display calibrating message
        cv2.putText(current_frame, "Keep a neutral face for calibration...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('face_landmarker', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    # Calculate the mean and standard deviation of the aspect ratios
    ear_mean, ear_std = np.mean(ear_values), np.std(ear_values)
    mar_mean, mar_std = np.mean(mar_values), np.std(mar_values)

    # Return the mean and standard deviation of the aspect ratios
    return ear_mean, ear_std, mar_mean, mar_std


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
        global DETECTION_RESULT

        DETECTION_RESULT = result

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
    ear_mean, ear_std, mar_mean, mar_std = calibrate(CALIBRATION_FRAME_COUNT, cap, detector)

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

        if DETECTION_RESULT:
            for face_landmarks in DETECTION_RESULT.face_landmarks:
                
                # Calculate the aspect ratios for the left and right eyes
                left_ear = eye_aspect_ratio(face_landmarks, left_eye_keypoints)
                right_ear = eye_aspect_ratio(face_landmarks, right_eye_keypoints)
                mar = mouth_aspect_ratio(face_landmarks, mouth_keypoints)

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

                # Check if the average aspect ratios are below the thresholds
                if np.mean(ear_history) < EAR_THRESHOLD:
                    print("Microsleep detected at time: ", time.asctime(time.localtime(time.time())), "with EAR: ", np.mean(ear_history))
                    cv2.putText(current_frame, "Microsleep detected!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if np.mean(mar_history) > MAR_THRESHOLD:
                    print("Yawning detected at time: ", time.asctime(time.localtime(time.time())), "with MAR: ", np.mean(mar_history))
                    cv2.putText(current_frame, "Yawning detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the aspect ratios on the image
                cv2.putText(current_frame, "Left eye aspect ratio: {:.2f}".format(left_ear), 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(current_frame, "Right eye aspect ratio: {:.2f}".format(right_ear), 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(current_frame, "Mouth aspect ratio: {:.2f}".format(mar), 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
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