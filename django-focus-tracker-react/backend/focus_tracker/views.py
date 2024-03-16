from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import DetectionEvent, Session
from .serializers import DetectionEventSerializer
from django.shortcuts import get_list_or_404
from rest_framework import status

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

class CurrentSessionView(APIView):
    def post(self, request, format=None):
        session_id = request.data.get('session_id')
        print("In current session view" + str(session_id))
        if session_id:
            # Create or update the session data here
            # For simplicity, this example just creates a new Session instance
            Session.objects.create(session_id=session_id)
            return Response({"status": "success", "session_id": session_id}, status=status.HTTP_201_CREATED)
        return Response({"status": "error", "message": "Missing session_id"}, status=status.HTTP_400_BAD_REQUEST)
    
    def get(self, request, *args, **kwargs):
        latest_session = Session.objects.order_by('-created_at').first()
        if latest_session:
            return Response({'session_id': latest_session.session_id})
        else:
            return Response({'status': 'error', 'message': 'No sessions available'}, status=404)


class YawningDataView(APIView):
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get('session_id')
        # session_id = request.data.session_id
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
