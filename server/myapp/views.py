# views.py
from PIL import Image, UnidentifiedImageError
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.conf import settings

from .utils import (
    save_uploaded_file_exact,
    predict_image,
)

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({'error': 'No image provided'}, status=400)

        # 1) Save to media/uploads/ with the ORIGINAL filename
        try:
            file_path = save_uploaded_file_exact(file_obj)
        except Exception as e:
            return Response({'error': f'Failed to save file: {e}'}, status=400)

        # 2) Open saved image and run prediction
        try:
            img = Image.open(file_path).convert('RGB')
        except UnidentifiedImageError:
            return Response({'error': 'Invalid image file'}, status=400)
        except Exception as e:
            return Response({'error': f'Failed to read image: {e}'}, status=400)

        try:
            predicted_class, confidence, lighting, brightness = predict_image(img)
            return Response({
                'prediction': predicted_class,
                'confidence': confidence,
                'lighting': lighting,
                # 'brightness': brightness,  # uncomment if you want it
                'image_url': settings.MEDIA_URL + "uploads/" + file_obj.name,
            })
        except Exception as e:
            return Response({'error': str(e)}, status=500)
