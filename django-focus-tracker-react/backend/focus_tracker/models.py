from django.db import models
from django.utils import timezone

class DetectionEvent(models.Model):
    session_id = models.CharField(max_length=255, null=True)
    user_id = models.CharField(max_length=255)
    detection_type = models.CharField(max_length=100)
    timestamp = models.DateTimeField(default=timezone.now)
    aspect_ratio = models.FloatField()

    def __str__(self):
        return f"{self.user_id} - {self.detection_type} at {self.timestamp}"
