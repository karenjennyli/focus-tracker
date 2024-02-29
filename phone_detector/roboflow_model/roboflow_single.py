from inference import get_roboflow_model
import supervision as sv
import cv2

model = get_roboflow_model(model_id='mobilephone-wusj2/5', api_key="mI2GwhI0NCN40uYZvW5d")

image = cv2.imread('test1.jpg')

results = model.infer(image)

detections = sv.Detections.from_inference(
    results[0].dict(by_alias=True, exclude_none=True)
)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [p.class_name for p in results[0].predictions]

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

sv.plot_image(annotated_image)
