from django.urls import path
from .views import DetectionEventView, DetectionDataView, CurrentSessionView, StartCalibration, EEGDataView, FlowDataView, FocusDataView, get_sessions, get_session_by_id, SessionHistoryDataView

urlpatterns = [
    path('api/detections/', DetectionEventView.as_view(), name='detection_events'),
    path('api/detection-data/', DetectionDataView.as_view(), name='detection_data'),
    path('api/current_session', CurrentSessionView.as_view(), name='current_session'),
    path('api/start_calibration/', StartCalibration, name='start_calibration'),
    path('api/eeg_data', EEGDataView.as_view(), name='eeg_data'),
    path('api/flow_data/', FlowDataView.as_view(), name='flow_data'),
    path('api/focus_data/', FocusDataView.as_view(), name='focus_data'),
    path('api/sessions', get_sessions, name='get_sessions'),
    path('api/session/<str:session_id>', get_session_by_id, name='get_session_by_id'),
    path('api/session_history', SessionHistoryDataView.as_view(), name='session_history_data'),
]
