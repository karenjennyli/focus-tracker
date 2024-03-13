'''
Performance:
3 fps

Can't use anymore because used up all free hosted API calls
'''

from roboflow import Roboflow
import supervision as sv
import cv2
import time

# Initialize the model
rf = Roboflow(api_key="mI2GwhI0NCN40uYZvW5d")
project = rf.workspace().project("mobilephone-wusj2")
model = project.version(5).model

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialize the time and frame count for FPS calculation
start_time = time.time()
frame_count = 0
fps_avg_frame_count = 10
counter = 0
fps = 0

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform inference on the frame
    result = model.predict(frame, confidence=40, overlap=30).json()

    labels = [item["class"] for item in result["predictions"]]

    detections = sv.Detections.from_roboflow(result)

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    if counter % fps_avg_frame_count == 0:
        fps = fps_avg_frame_count / (time.time() - start_time)
        start_time = time.time()
    
    counter += 1

    # calculate the frame per second and draw it on the frame
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', annotated_image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()
