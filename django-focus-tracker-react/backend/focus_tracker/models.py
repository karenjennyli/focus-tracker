from django.db import models
from django.utils import timezone

class Session(models.Model):
    session_id = models.CharField(max_length=255, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

class DetectionEvent(models.Model):
    session_id = models.CharField(max_length=255, null=True)
    user_id = models.CharField(max_length=255)
    detection_type = models.CharField(max_length=100)
    timestamp = models.DateTimeField(default=timezone.now)
    image = models.ImageField(upload_to='detectionimages/', null=True, blank=True)

    def __str__(self):
        return f"{self.user_id} - {self.detection_type} at {self.timestamp}"
