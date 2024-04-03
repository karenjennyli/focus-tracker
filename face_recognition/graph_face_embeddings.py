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

from deepface import DeepFace
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition

import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Result of the face landmark detection
DETECTION_RESULT = None

# Calculate FPS
FPS_AVG_FRAME_COUNT = 10
COUNTER, FPS = 0, 0
START_TIME = time.time()

# Keypoints on the face
KEYPOINTS = [10, 152, 127, 356]

# Detector backend
DETECTOR_BACKEND = 'fastmtcnn'
MODEL_NAME = 'SFace'
DISTANCE_METRIC = 'euclidean_l2'

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
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    landmark_list.landmark.extend([face_landmarks_proto.landmark[i] for i in KEYPOINTS])
    mp_drawing.draw_landmarks(
        image=current_frame,
        landmark_list=landmark_list,
        connections=[],
        landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        connection_drawing_spec=None
    )

def graph_distance(img1_representation, img2_representation, current_distance, threshold, img1, img2, time):
    img1_graph = []
    img2_graph = []

    for i in range(0, 200):
        img1_graph.append(img1_representation)
        img2_graph.append(img2_representation)

    img1_graph = np.array(img1_graph)
    img2_graph = np.array(img2_graph)

    # ----------------------------------------------
    # plotting

    fig = plt.figure()

    ax1 = fig.add_subplot(3, 2, 1)
    plt.imshow(img1[0])
    plt.axis("off")

    ax2 = fig.add_subplot(3, 2, 2)
    im = plt.imshow(img1_graph, interpolation="nearest", cmap=plt.cm.ocean)
    plt.colorbar()

    ax3 = fig.add_subplot(3, 2, 3)
    plt.imshow(img2[0])
    plt.axis("off")

    ax4 = fig.add_subplot(3, 2, 4)
    im = plt.imshow(img2_graph, interpolation="nearest", cmap=plt.cm.ocean)
    plt.colorbar()

    ax5 = fig.add_subplot(3, 2, 5)
    plt.text(0.35, 0, f"Distance: {current_distance}")
    plt.text(0.35, 0.5, f"Threshold: {threshold}")
    plt.text(0.35, 1, f"Time: {time}")
    plt.axis("off")

    plt.show()

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
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def save_result(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % FPS_AVG_FRAME_COUNT == 0:
            FPS = FPS_AVG_FRAME_COUNT / (time.time() - START_TIME)
            print('FPS:', FPS)
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

    # Build face recognition model
    recognition_model: FacialRecognition = DeepFace.build_model(model_name=MODEL_NAME)
    target_size = recognition_model.input_shape

    # Get template face image
    template = cv2.imread('dataset/karen3.jpg')
    template = cv2.flip(template, 1)
    template = np.array(template)
    template_faces = DeepFace.extract_faces(template, target_size=target_size, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
    if len(template_faces) == 0:
        sys.exit("No face detected in the template image")
    else:
        template = template_faces[0]["face"]
    template = np.expand_dims(template, axis=0)  # to (1, 224, 224, 3)
    template_embedding = recognition_model.find_embeddings(template)
    template_embedding = np.array(template_embedding)
    threshold = verification.find_threshold(model_name=MODEL_NAME, distance_metric=DISTANCE_METRIC)

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

        # Image width and height
        img_height, img_width, _ = current_frame.shape

        if DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
            face_landmarks = DETECTION_RESULT.face_landmarks[0]

            top_landmark = face_landmarks[10]
            bottom_landmark = face_landmarks[152]
            left_landmark = face_landmarks[127]
            right_landmark = face_landmarks[356]

            top = int(top_landmark.y * img_height)
            bottom = int(bottom_landmark.y * img_height)
            left = int(left_landmark.x * img_width)
            right = int(right_landmark.x * img_width)

            # increase bounding box size
            width_factor = 0.5
            height_factor = 0.5
            width_increase = int((right - left) * width_factor)
            height_increase = int((bottom - top) * height_factor)
            left = max(0, left - width_increase)
            right = min(img_width, right + width_increase)
            top = max(0, top - height_increase)
            bottom = min(img_height, bottom + height_increase)

            top_left = (left, top)
            top_right = (right, top)
            bottom_left = (left, bottom)
            bottom_right = (right, bottom)

            start_time = time.time()

            # extract the face
            face = current_frame[top:bottom, left:right]
            face = np.array(face)
            faces = DeepFace.extract_faces(face, target_size=target_size, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
            # import pdb; pdb.set_trace()

            if len(faces) == 0:
                print("No face detected")
                continue
            else:
                face = faces[0]["face"]
            face = np.expand_dims(face, axis=0)  # to (1, 224, 224, 3)
            face_embedding = recognition_model.find_embeddings(face)
            face_embedding = np.array(face_embedding)

            # compute the distance between the embeddings
            distance = verification.find_distance(face_embedding, template_embedding, distance_metric=DISTANCE_METRIC)
            print(distance < threshold, distance, threshold, time.time() - start_time)

            graph_distance(template_embedding, face_embedding, distance, threshold, template, face, time.time() - start_time)
            
            # draw the bounding box around the face
            cv2.line(current_frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(current_frame, bottom_left, bottom_right, (0, 255, 0), 2)
            cv2.line(current_frame, top_left, bottom_left, (0, 255, 0), 2)
            cv2.line(current_frame, top_right, bottom_right, (0, 255, 0), 2)
            draw_face_landmarks(current_frame, face_landmarks)

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
        default=1080)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=1920)
    args = parser.parse_args()

    run(args.model, int(args.numFaces), args.minFaceDetectionConfidence,
        args.minFacePresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()
