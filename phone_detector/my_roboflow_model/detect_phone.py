# https://supervision.roboflow.com/latest/how_to/detect_and_annotate/

from inference import get_roboflow_model
import supervision as sv
import cv2
import time

# Initialize the model
model = get_roboflow_model(model_id='phone-3ekgi/2', api_key='g2xhIxwutQ35mvQxltNJ')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
percentage_bar_annotator = sv.PercentageBarAnnotator()

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
    results = model.infer(frame, confidence=0.2)

    # Convert the results to detections
    detections = sv.Detections.from_inference(
        results[0].dict(by_alias=True, exclude_none=True)
    )

    # Get the labels
    labels = [p.class_name for p in results[0].predictions]

    # Annotate the frame with bounding boxes and labels
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = percentage_bar_annotator.annotate(
        scene=annotated_image, detections=detections
    )

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
