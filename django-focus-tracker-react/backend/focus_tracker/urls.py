from django.urls import path
from .views import DetectionEventView

urlpatterns = [
    path('api/detections/', DetectionEventView.as_view(), name='detection_events'),
]