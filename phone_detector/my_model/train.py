# https://docs.ultralytics.com/modes/train/#train-settings

from clearml import Task
from ultralytics import YOLO

task =  Task.init(project_name="phone-detector", task_name="train")

model_variant = "yolov8n"
task.set_parameter("model_variant", model_variant)

model = YOLO(f'{model_variant}.pt')

args = dict(data='data.yaml', epochs=100, imgsz=640, device='mps', verbose=True, plots=True)
task.connect(args)

results = model.train(**args)
