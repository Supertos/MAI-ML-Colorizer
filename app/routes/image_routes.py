from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.services.db_service import get_db, ImageModel, create_image_record, get_image_by_id, update_image_status_and_result
from app.services.s3_service import upload_file_to_s3, get_s3_file_url
from celery_app.tasks import process_image_task

router = APIRouter()

@router.post("/")
async def upload_image(
    file: UploadFile = File(...),
    grain: int = Form(...),
    sharpness: int = Form(...)
):
    #принимает изображение и параметры, сохраняет данные в S3 и БД.
    # cоздаёт Celery-задачу на обработку

    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Файл не является изображением")

    image_bytes = await file.read()
    original_key = await upload_file_to_s3(image_bytes, file.filename)

    db: Session = next(get_db())
    new_image = create_image_record(
        db=db,
        original_filename=file.filename,
        s3_path=original_key,
        grain=grain,
        sharpness=sharpness
    )

    # Отправляем задачу Celery (без .apply, чтобы не блокировать)
    process_image_task.delay(new_image.id)

    return JSONResponse(content={"image_id": new_image.id})

@router.get("/{image_id}")
async def get_image_status(image_id: int):
    
    db: Session = next(get_db())
    image_obj = get_image_by_id(db, image_id)
    if not image_obj:
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    response_data = {
        "id": image_obj.id,
        "status": image_obj.status,
        "original_file": get_s3_file_url(image_obj.s3_path) if image_obj.s3_path else None,
        "processed_file": get_s3_file_url(image_obj.processed_s3_path) if image_obj.processed_s3_path else None
    }

    return JSONResponse(content=response_data)
