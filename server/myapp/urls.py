from django.urls import path
from .views import PredictView
#from .genai import Detect, VideoStreamDetectionView, VideoStreamMJPEGView

urlpatterns = [
    path('predict-classification', PredictView.as_view(), name='predict'), #classification endpoint
    # path('detect_damage/', Detect.as_view(), name='detect_damage'),# this is the detection endpoint with classification results
    # path("detect/", VideoStreamDetectionView.as_view(), name="detect"), # captured images frame from the stream
    # path('live-stream/', VideoStreamMJPEGView.as_view(), name='live_stream'),#this is the live stream endpoint
]
