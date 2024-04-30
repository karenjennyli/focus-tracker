from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import DetectionEvent, Session, EEGEvent, FlowEvent, FocusEvent, SessionHistoryEvent, SessionLength
from .serializers import DetectionEventSerializer, EEGEventSerializer, FlowEventSerializer, FocusEventSerializer, SessionHistoryEventSerializer, SessionLengthSerializer
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
            return Response({'session_id': latest_session.session_id, 'created_at': latest_session.created_at})
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
    venv_path = os.environ.get('VENV_PATH')
    focus_tracker_dir = os.environ.get('FOCUS_TRACKER_DIR')
    working_dir = focus_tracker_dir + '/video_processing'
    script_path = focus_tracker_dir + '/video_processing/run.py'
    if not working_dir or not script_path:
        return JsonResponse({"error": "Environment variables for script path or working directory not set"}, status=500)

    try:
        command = f'''osascript -e 'tell app "Terminal" to do script "source {venv_path} && cd {working_dir} && python3 {script_path}"' '''
        subprocess.Popen(command, shell=True)
        return JsonResponse({"message": "Calibration started successfully"}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
    
class EEGDataView(APIView):
    def post(self, request, format=None):
        print("Received POST data:", request.data)
        serializer = EEGEventSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            # print the validated data
            # print("Saved DetectionEvent:", serializer.validated_data)
            return Response(serializer.data, status=201)
        else:
            print("not valid")

        print(serializer.data)
        return Response(serializer.errors, status=400)
    def get(self, request, *args, **kwargs):
        eeg_data = EEGEvent.objects.order_by('-timestamp_formatted')
        serializer = EEGEventSerializer(eeg_data, many=True)
        return Response(serializer.data)

class FlowDataView(APIView):
    def post(self, request, format=None):
        print("Received Flow POST data:", request.data)
        serializer = FlowEventSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        else:
            print("not valid")

        print(serializer.data)
        return Response(serializer.errors, status=400)
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get('session_id')
        if session_id:
            eeg_data = FlowEvent.objects.filter(session_id=session_id).order_by('-timestamp_formatted')
        else:
            eeg_data = get_list_or_404(FlowEvent)
        serializer = FlowEventSerializer(eeg_data, many=True)
        return Response(serializer.data)
    
class FocusDataView(APIView):
    def post(self, request, format=None):
        serializer = FocusEventSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        else:
            print("not valid")

        print(serializer.data)
        return Response(serializer.errors, status=400)
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get('session_id')
        if session_id:
            eeg_data = FocusEvent.objects.filter(session_id=session_id).order_by('-timestamp_formatted')
        else:
            eeg_data = get_list_or_404(FocusEvent)
        serializer = FocusEventSerializer(eeg_data, many=True)
        return Response(serializer.data)

def get_sessions(request):
    sessions = Session.objects.all().values('session_id', 'created_at')  # Get necessary fields
    session_list = list(sessions)  # Convert QuerySet to a list of dicts
    print(session_list)
    return JsonResponse(session_list, safe=False)  # Return as JSON

def get_session_by_id(request, session_id):
    try:
        session = Session.objects.get(session_id=session_id)
        return JsonResponse({
            'session_id': session.session_id,
            'created_at': session.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        })
    except Session.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)

class SessionHistoryDataView(APIView):
    # def post(self, request, format=None):
    #     print("Received POST data:", request.data)
    #     serializer = SessionHistoryEventSerializer(data=request.data, context={'request': request})
    #     if serializer.is_valid():
    #         serializer.save()
    #         # print the validated data
    #         # print("Saved DetectionEvent:", serializer.validated_data)
    #         return Response(serializer.data, status=201)
    #     return Response(serializer.errors, status=400)

    def post(self, request, format=None):
        print("Received POST data:", request.data)
        session_id = request.data.get('session_id')
        total_distractions = request.data.get('total_distractions', 0)

        # Try to retrieve an existing session history record
        session_history, created = SessionHistoryEvent.objects.get_or_create(
            session_id=session_id,
            defaults={'total_distractions': total_distractions}
        )

        # If the record already exists and isn't just created, update it
        if not created:
            session_history.total_distractions = total_distractions
            session_history.save()

        # Serialize the record to return updated data
        serializer = SessionHistoryEventSerializer(session_history, context={'request': request})
        
        # Choose the response status based on whether the record was created or updated
        response_status = status.HTTP_201_CREATED if created else status.HTTP_200_OK

        return Response(serializer.data, status=response_status)
    def get(self, request, *args, **kwargs):
        session_history_data = SessionHistoryEvent.objects.all()
        serializer = SessionHistoryEventSerializer(session_history_data, many=True)
        return Response(serializer.data)

class SessionLengthDataView(APIView):
    def post(self, request, format=None):
        serializer = SessionLengthSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        else:
            print("not valid")

    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get('session_id')
        # print all session length data
        print(SessionLength.objects.all())
        session_length_data = SessionLength.objects.filter(session_id=session_id)
        serializer = SessionLengthSerializer(session_length_data, many=True)
        return Response(serializer.data)
