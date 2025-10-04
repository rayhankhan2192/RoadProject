from django.urls import path
from .views import PredictView, DetectView

urlpatterns = [
    path('predict-classification', PredictView.as_view(), name='predict'), #classification endpoint
    path('detect_damage', DetectView.as_view(), name='detect_damage'),# this is the detection endpoint with classification results
]
