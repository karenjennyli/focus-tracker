from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import DetectionEvent
from .serializers import DetectionEventSerializer
from django.shortcuts import get_list_or_404

class DetectionEventView(APIView):
    def post(self, request, format=None):
        # Print the raw POST data
        print("Received POST data:", request.data)

        serializer = DetectionEventSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            # print the validated data
            print("Saved DetectionEvent:", serializer.validated_data)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

class YawningDataView(APIView):
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get('session_id')
        print("HERE" + str(session_id))
        if session_id:
            print("here")
            yawning_data = DetectionEvent.objects.filter(session_id=session_id, detection_type='yawn').order_by('-timestamp')
            print(yawning_data)
        else:
            yawning_data = get_list_or_404(DetectionEvent, detection_type='yawn')
        serializer = DetectionEventSerializer(yawning_data, many=True)
        return Response(serializer.data)
    
# class YawningDataView(APIView):
#     def get(self, request, *args, **kwargs):
#         session_id = request.query_params.get('session_id', None)
#         if session_id is not None:
#             yawning_data = DetectionEvent.objects.filter(detection_type='yawn', session_id=session_id)
#         else:
#             yawning_data = DetectionEvent.objects.none()  # Return no data if no sessionId is provided
#         serializer = DetectionEventSerializer(yawning_data, many=True)
#         return Response(serializer.data)
