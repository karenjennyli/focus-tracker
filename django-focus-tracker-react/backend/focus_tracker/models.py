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
    aspect_ratio = models.FloatField()
    image = models.ImageField(upload_to='detectionimages/', null=True, blank=True)
    frequency = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user_id} - {self.detection_type} at {self.timestamp}"
    
class EEGEvent(models.Model):
    # Assuming 'timestamp_epoch' is a UNIX epoch time, store as an IntegerField
    timestamp_epoch = models.FloatField(help_text="Epoch timestamp of the event")

    # 'timestamp_formatted' might be redundant to store in the database since
    # it can be derived from 'timestamp_epoch', but if you want to cache the
    # formatted string, use a CharField
    timestamp_formatted = models.CharField(max_length=8, help_text="Formatted time as HH:MM:SS")

    # Assuming 'focus_pm' represents a measurement or value related to the event,
    # the data type could vary but here it's stored as a FloatField
    focus_pm = models.FloatField(help_text="Focus measurement")

    # Adding auto-added created and modified timestamps for record-keeping
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        # Customize the string representation of the model, for example:
        return f"Event at {self.timestamp_formatted}"
    
class FlowEvent(models.Model):
    session_id = models.CharField(max_length=255, null=True)
    timestamp_epoch = models.FloatField(help_text="Epoch timestamp of the event")
    timestamp_formatted = models.CharField(max_length=8, help_text="Formatted time as HH:MM:SS")

    flow = models.CharField(max_length=20, help_text="Flow State (Not in Flow or Flow)", default="")
    flowCount = models.IntegerField(default=0)
    notInFlowCount = models.IntegerField(default=0)
    predictionSum = models.IntegerField(default=0)
    flowNotFlowRatio = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        # Customize the string representation of the model, for example:
        return f"Event at {self.timestamp_formatted}"

class FocusEvent(models.Model):
    session_id = models.CharField(max_length=255, null=True)
    timestamp_epoch = models.FloatField(help_text="Epoch timestamp of the event")
    timestamp_formatted = models.CharField(max_length=8, help_text="Formatted time as HH:MM:SS")

    focus = models.CharField(max_length=20, help_text="Focus State (Not in Focus or Focus)", default="")
    focusCount = models.IntegerField(default=0)
    notInFocusCount = models.IntegerField(default=0)
    predictionSum = models.IntegerField(default=0)
    FocusNotFocusRatio = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        # Customize the string representation of the model, for example:
        return f"Event at {self.timestamp_formatted}"

class SessionHistoryEvent(models.Model):
    session_id = models.CharField(max_length=255, null=True)
    total_distractions = models.IntegerField(default=0)
    timestamp = models.DateTimeField(default=timezone.now)

class SessionLength(models.Model):
    session_id = models.CharField(max_length=255, null=True)
    session_length = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(default=timezone.now)
