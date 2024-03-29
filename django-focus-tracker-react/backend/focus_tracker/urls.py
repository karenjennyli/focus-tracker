from django.urls import path
from .views import DetectionEventView, DetectionDataView, CurrentSessionView, StartCalibration

urlpatterns = [
    path('api/detections/', DetectionEventView.as_view(), name='detection_events'),
    path('api/detection-data/', DetectionDataView.as_view(), name='detection_data'),
    path('api/current_session', CurrentSessionView.as_view(), name='current_session'),
    path('api/start_calibration/', StartCalibration, name='start_calibration'),
]