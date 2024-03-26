from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import DetectionEvent, Session
from .serializers import DetectionEventSerializer
from django.shortcuts import get_list_or_404
from rest_framework import status
from django.core.files.base import ContentFile
import base64
from django.utils.timezone import now
import subprocess
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os

class DetectionEventView(APIView):
    def post(self, request, format=None):
        print("Received POST data:", request.data)
        # Decode image if present
        if 'image' in request.data:
            imgstr = request.data['image']  # Direct base64 data
            filename = f"{now().strftime('%Y%m%d%H%M%S')}.jpg"
            data = ContentFile(base64.b64decode(imgstr), name=filename)
            request.data['image'] = data

        serializer = DetectionEventSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            # print the validated data
            # print("Saved DetectionEvent:", serializer.validated_data)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

class CurrentSessionView(APIView):
    def post(self, request, format=None):
        session_id = request.data.get('session_id')
        # print("In current session view" + str(session_id))
        if session_id:
            Session.objects.create(session_id=session_id)
            return Response({"status": "success", "session_id": session_id}, status=status.HTTP_201_CREATED)
        return Response({"status": "error", "message": "Missing session_id"}, status=status.HTTP_400_BAD_REQUEST)
    
    def get(self, request, *args, **kwargs):
        latest_session = Session.objects.order_by('-created_at').first()
        if latest_session:
            return Response({'session_id': latest_session.session_id})
        else:
            return Response({'status': 'error', 'message': 'No sessions available'}, status=404)


class DetectionDataView(APIView):
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get('session_id')
        # print("HERE" + str(session_id))
        if session_id:
            detection_data = DetectionEvent.objects.filter(session_id=session_id).order_by('-timestamp')
        else:
            detection_data = get_list_or_404(DetectionEvent)
        serializer = DetectionEventSerializer(detection_data, many=True)
        return Response(serializer.data)

@csrf_exempt
def StartCalibration(request):
    # Access environment variables
    working_dir = os.environ.get('SCRIPT_WORKING_DIR')
    script_path = os.environ.get('SCRIPT_PATH')
    # working_dir = '/Users/arnavarora/Documents/focus-tracker/video_processing'
    # script_path = '/Users/arnavarora/Documents/focus-tracker/video_processing/run.py'
    if not working_dir or not script_path:
        return JsonResponse({"error": "Environment variables for script path or working directory not set"}, status=500)

    try:
        subprocess.Popen(['python', script_path], cwd=working_dir)
        return JsonResponse({"message": "Calibration started successfully"}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    