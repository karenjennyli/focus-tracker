from django.urls import path
from .views import DetectionEventView, DetectionDataView, CurrentSessionView, EEGDataView

urlpatterns = [
    path('api/detections/', DetectionEventView.as_view(), name='detection_events'),
    path('api/detection-data/', DetectionDataView.as_view(), name='detection_data'),
    path('api/current_session', CurrentSessionView.as_view(), name='current_session'),
    path('api/eeg_data', EEGDataView.as_view(), name='eeg_data'),
]
