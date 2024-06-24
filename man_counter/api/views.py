import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import CreateAPIView
from .models import Image
from .serializers import ImageSerializer
import yolov9
import os
from django.conf import settings
from man_counter.settings import MEDIA_ROOT, YOLO_PATH
import sys
import cv2

# Load the YOLO model once and reuse it
def load_yolo_model():
    model = yolov9.load(
        YOLO_PATH,
        device="cpu",
    )
    model.conf = 0.5  # NMS confidence threshold
    model.iou = 0.7  # NMS IoU threshold
    model.classes = [0]
    return model

# Process image using the loaded YOLO model
def process_image(model, path, size):
    results = model(path, size=size)
    return results

class ImageUploadView(CreateAPIView):
    serializer_class = ImageSerializer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = load_yolo_model()

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        if response.status_code == status.HTTP_201_CREATED:
            image_url = response.data['image']
            local_image_path = os.path.join(settings.MEDIA_ROOT, os.path.basename(image_url))

            # Download the image from the URL to the local path
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(local_image_path, 'wb') as f:
                    f.write(response.content)
            else:
                return Response({'error': 'Failed to download image'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            if not os.path.exists(local_image_path):
                return Response({'error': 'File not found'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            try:
                image = cv2.imread(local_image_path)
                height, width, _ = image.shape
            except:
                height, width = 640, 640    

            results = process_image(self.model, local_image_path, size=(height, width))
            simplified_results = simplify_results(results)

            processed_data = {
                'count': simplified_results
            }
            os.remove(local_image_path)
            return Response(processed_data, status=status.HTTP_200_OK)
        return response

def simplify_results(results):
    # Simplify the results to return only the count of detected people
    count = 0
    for result in results.xyxy:
        for detection in result:
            if int(detection[5]) == 0:  # Assuming class 0 is the person class
                count += 1
    return count