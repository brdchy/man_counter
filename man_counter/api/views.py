import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import CreateAPIView
from .serializers import ImageSerializer
import yolov9
import os
from django.conf import settings
from man_counter.settings import MEDIA_ROOT, YOLO_PATH
import sys
import cv2
import numpy as np
from io import BytesIO
import base64

# Load the YOLO model once and reuse it
def load_yolo_model():
    model = yolov9.load(
        YOLO_PATH,
        device="cpu",
    )
    model.conf = 0.4  # NMS confidence threshold
    model.iou = 0.7  # NMS IoU threshold
    model.classes = [0]
    return model

# Process image using the loaded YOLO model
def process_image(model, image, size):
    results = model(image, size=size)
    return results

class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        # Проверяем, есть ли файл в запросе
        if 'image' not in request.FILES:
            return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Получаем файл из запроса
        image_file = request.FILES['image']

        try:
            # Чтение изображения из файла
            image_data = image_file.read()
            image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return Response({'error': 'Invalid image file'}, status=status.HTTP_400_BAD_REQUEST)

            height, width, _ = image.shape

            # Обработка изображения с помощью YOLO
            model = load_yolo_model()
            results = process_image(model, image, size=(height, width))
            count, annotated_image = simplify_results(results, image)

            # Кодируем обработанное изображение в base64
            _, img_encoded = cv2.imencode('.jpg', annotated_image)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            processed_data = {
                'count': count,
                'image': img_base64
            }

            return Response(processed_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

def simplify_results(results, image):
    # Simplify the results to return only the count of detected people
    count = 0
    for result in results.xyxy:
        for detection in result:
            if int(detection[5]) == 0:  # Assuming class 0 is the person class
                count += 1
                x1, y1, x2, y2 = map(int, detection[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return count, image