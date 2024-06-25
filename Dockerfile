# Используйте официальный образ Python как базовый
FROM python:3.10
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
# Установите рабочую директорию внутри контейнера
WORKDIR /app

# Копируйте requirements.txt в контейнер
COPY requirements.txt .
RUN pip3 install --upgrade pip
# Установите зависимости
RUN pip3 install --no-cache-dir -r requirements.txt
# Копируйте остальные файлы проекта в контейнер
COPY . .
#RUN curl https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt -o yolov9e.pt

RUN wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt -O  man_counter/yolov9e1.pt
#RUN mv yolov9-e-converted.pt yolov9e.pt
RUN python man_counter/manage.py makemigrations
RUN python man_counter/manage.py migrate
# Команду для запуска приложения
CMD ["python", "man_counter/manage.py", "runserver", "0.0.0.0:8000"]
