import io
import os
import uuid
from celery import Celery
from PIL import Image, ImageOps

BROKER_URL = "amqp://guest:guest@localhost:5672//"
BACKEND_URL = "rpc://"

celery_app = Celery("tasks", broker=BROKER_URL, backend=BACKEND_URL)

# папка, куда кладём готовые файлы
PROCESSED_FOLDER = os.path.join("app", "static", "processed")
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@celery_app.task
def process_image_task(*args) -> str:
    #инвертируем картинку
    return None
