from django.urls import path
from .views import DetectionEventView, YawningDataView, CurrentSessionView

urlpatterns = [
    path('api/detections/', DetectionEventView.as_view(), name='detection_events'),
    path('api/yawning-data/', YawningDataView.as_view(), name='yawning_data'),
    path('api/current_session', CurrentSessionView.as_view(), name='current_session'),
]