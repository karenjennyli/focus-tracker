from django.urls import path
from .views import DetectionEventView, YawningDataView

urlpatterns = [
    path('api/detections/', DetectionEventView.as_view(), name='detection_events'),
    path('api/yawning-data/', YawningDataView.as_view(), name='yawning_data'),
]