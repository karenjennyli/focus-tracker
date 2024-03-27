from rest_framework import serializers
from .models import DetectionEvent, EEGEvent
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
