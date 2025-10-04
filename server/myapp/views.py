from PIL import Image, UnidentifiedImageError
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.conf import settings
import os


from .utils import (
    save_uploaded_file_exact,
    predict_image,
    normalize_single_line,
)
from .generative import generate_road_damage_report


class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image = request.FILES.get('image')
        if not image:
            return Response({'error': 'No image provided'}, status=400)

        # 1) Save to media/uploads/ with the ORIGINAL filename
        try:
            file_path = save_uploaded_file_exact(image)
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
                'image_url': settings.MEDIA_URL + "uploads/" + image.name,
            })
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class DetectView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        image = request.FILES.get("image")
        if not image:
            return Response({"error": "Image file not provided."}, status=400)

        # Save uploaded file
        try:
            file_path = save_uploaded_file_exact(image)
        except Exception as e:
            return Response({"error": f"Failed to save file: {e}"}, status=400)

        # Validate image
        try:
            img = Image.open(file_path).convert("RGB")
        except UnidentifiedImageError:
            return Response({"error": "Invalid image file"}, status=400)
        except Exception as e:
            return Response({"error": f"Failed to read image: {e}"}, status=400)

        # Local prediction
        try:
            predicted_class, confidence, lighting, brightness = predict_image(img)
        except Exception as e:
            return Response({"error": f"Prediction failed: {e}"}, status=500)

        # Generate everything in one go
        try:
            result = generate_road_damage_report(file_path)
        except Exception as e:
            return Response({"error": f"Failed to generate report: {e}"}, status=500)

        # Normalize text fields
        result["prompt"] = normalize_single_line(result["prompt"])
        result["report"] = normalize_single_line(result["report"])

        # URLs
        image_original_url = os.path.join(settings.MEDIA_URL, "uploads", image.name)
        image_detected_url = os.path.join(settings.MEDIA_URL, "detected", result["detected_filename"])

        # Combine and return
        return Response(
            {
                "prediction": {
                    "class": predicted_class,
                    "confidence": confidence,
                    "lighting": lighting,
                },
                "road_damage": result,
                "image_original": image_original_url,
                "image_detected": image_detected_url,
            },
            status=200,
        )