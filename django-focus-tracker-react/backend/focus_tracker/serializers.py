from rest_framework import serializers
from .models import DetectionEvent, EEGEvent, FlowEvent, FocusEvent, SessionHistoryEvent
from django.conf import settings

class DetectionEventSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = DetectionEvent
        fields = '__all__'
    
    def get_image_url(self, obj):
        request = self.context.get('request')
        if obj.image and hasattr(obj.image, 'url'):
            # Build the full URL
            return request.build_absolute_uri(obj.image.url) if request else obj.image.url
        else:
            # return None if there is no image
            return None
       
class EEGEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = EEGEvent
        fields = '__all__'

class FlowEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = FlowEvent
        fields = '__all__'

class FocusEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = FocusEvent
        fields = '__all__'

class SessionHistoryEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = SessionHistoryEvent
        fields = '__all__'
