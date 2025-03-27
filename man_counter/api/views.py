import cv2
import numpy as np
import base64
from io import BytesIO
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import yolov9
from man_counter.settings import MEDIA_ROOT, YOLO_PATH

# Загрузка модели YOLO один раз при старте приложения
YOLO_MODEL = yolov9.load(
    YOLO_PATH,  # Укажите правильный путь к модели
    device="0",         # Используйте "cuda", если есть GPU
)
YOLO_MODEL.conf = 0.4     # Порог уверенности для детекции
YOLO_MODEL.iou = 0.7      # Порог IoU для NMS
YOLO_MODEL.classes = [0]  # Класс 0 (обычно "человек" в COCO)

class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        # Проверяем, есть ли файл в запросе
        if 'image' not in request.FILES:
            return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']

        try:
            # Чтение изображения
            image_data = image_file.read()
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return Response({'error': 'Invalid image file'}, status=status.HTTP_400_BAD_REQUEST)

            height, width, _ = image.shape

            # Детекция людей с помощью YOLO
            results = YOLO_MODEL(image, size=(height, width))
            count, annotated_image = self._simplify_results(results, image)

            # Кодируем изображение в base64
            _, img_encoded = cv2.imencode('.jpg', annotated_image)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            return Response({
                'count': count,
                'image': img_base64,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _simplify_results(self, results, image):
        """Обрабатывает результаты YOLO: подсчёт людей и рисование bounding boxes."""
        count = 0
        for result in results.xyxy:
            for detection in result:
                if int(detection[5]) == 0:  # Класс 0 = человек
                    count += 1
                    x1, y1, x2, y2 = map(int, detection[:4])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return count, image