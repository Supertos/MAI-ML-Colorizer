**Проект Has-Been: Восстановление цвета фотографий**

**Описание**

Проект состоит из:

* **Backend** на FastAPI + Celery для асинхронной обработки и очередей.
* **Frontend** на React + Vite для удобного интерфейса загрузки и просмотра результатов.
* **Хранилище**: MinIO (S3-совместимое).
* **Очередь задач**: RabbitMQ.
* **База данных**: PostgreSQL.

**Основные возможности**

* Загрузка одной или нескольких фотографий.
* Асинхронная раскраска через Celery + модель TensorFlow.
* Хранение изображений в MinIO и выдача URL-ов.
* Архивация и скачивание всех результатов одним ZIP-файлом.

---

##  Архитектура проекта

```
project-root/
├── backend/                                            # FastAPI + Celery + модели
│   ├── app/
│   │   ├── main.py                                     # Эндпоинты /upload, /images
│   │   ├── celery_app.py                               # Инициализация Celery
│   │   ├── tasks.py                                    # Celery-таски (раскраска)
│   │   ├── s3_utils.py                                 # Работа с MinIO
│   │   ├── 5_128_2048_checkpoint.weights.h5            #веса
│   │   ├── database.py                                 # SQLAlchemy + сессии
│   │   ├── models.py                                   # Модель Image
│   │   └── model_class/                                # Реализация build_colorizer
│   ├── Dockerfile
│   ├── requirements.txt                                # Зависимости Python
│   ├── alembic/                                        # миграции постгреса
│   └── .env  
├── frontend/               # React + Vite
│   ├── src/
│   ├── vite.config.js
│   ├── package.json
│   └── ...
└── docker-compose.yml      # Описание всех сервисов
```

---

##  Технологии и зависимости

* **Backend**: Python 3.10, FastAPI, Uvicorn, Celery, SQLAlchemy, Alembic, PostgreSQL, RabbitMQ, MinIO, Boto3, Pillow, OpenCV, TensorFlow/Keras.
* **Frontend**: React, React Router, Vite, TailwindCSS (или Sass), JavaScript/TypeScript.
* **Docker**: Docker Engine, Docker Compose.

---

##  Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Supertos/MAI-ML-Colorizer.git
cd project-root
```

### 2. Запуск с помощью Docker Compose

```bash
docker-compose up -d --build
```

* FastAPI API на `http://localhost:8000`
* Nginx-прокси на `http://localhost:8080`
* RabbitMQ UI на `http://localhost:15672` (`guest`/`guest`)
* MinIO Console на `http://localhost:9001` (`minioadmin`/`minioadmin`)

### 3. Применить миграции Alembic

```bash
docker-compose exec backend_fastapi alembic upgrade head
```

---

## API

### `POST /upload`

* **Описание**: загрузка одного файла и запуск таска.
* **Параметры**:

  * `file`: файл изображения (PNG, JPEG).
  * `grain`, `sharpness`: числа 0–100 (не используются!).
  * `anonymous_id`: строка (необязательно), возвращается и сохраняется в cookie.
* **Ответ**:

```json
{
  "image_id": 1,
  "original_filename": "photo.jpg",
  "s3_key": "<uuid>_photo.jpg",
  "anonymous_id": "<uuid>"
}
```

### `GET /images/{anonymous_id}`

* **Описание**: получить список записей пользователя.
* **Ответ**: массив объектов:

```json
[
  {
    "id": 1,
    "filename": "photo.jpg",
    "s3_key": "...",
    "created_at": "2025-05-28T...Z",
    "download_url": "presigned_url",
    "inverted_download_url": "presigned_url_for_colorized"
  }
]
```

---

##  Frontend

1. Перейдите в папку `frontend/`.
2. Установите зависимости:

   ```bash
   npm install
   ```
3. Запустите dev-сервер Vite:

   ```bash
   npm run dev
   ```
4. Откройте `http://localhost:3000`.

Frontend сам сохраняет `anonymous_id` в cookie, последующие запросы идут правильно.

---
