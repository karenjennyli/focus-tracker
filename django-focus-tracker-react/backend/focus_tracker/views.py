from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import DetectionEvent
from .serializers import DetectionEventSerializer

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
