import os
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from app.services.tasks import process_image_task, PROCESSED_FOLDER
import os

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def index_page():
    """
    Простейшая HTML-форма загрузки.
    """
    return """
    <!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Upload with Grain & Sharpness</title>
</head>
<body>
    <h1>Загрузка файла + параметры</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <p>
            <label for="file">Выберите файл:</label><br>
            <input type="file" id="file" name="file" accept="image/*" required>
        </p>
        <p>
            <label for="grain">Зернистость:</label><br>
            <input type="number" id="grain" name="grain" value="50" min="0" max="100">
        </p>
        <p>
            <label for="sharpness">Чёткость:</label><br>
            <input type="number" id="sharpness" name="sharpness" value="50" min="0" max="100">
        </p>
        <button type="submit">Загрузить</button>
    </form>
</body>
</html>

    """

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    grain: int = Form(...),
    sharpness: int = Form(...)
):
   return None

@router.get("/show/{filename}", response_class=HTMLResponse)
def show_processed_image(filename: str):
    return None

