from celery import Celery
from celery.utils.log import get_task_logger
from PIL import Image, ImageOps, ImageEnhance
import io

from app import config
from app.services.db_service import SessionLocal, get_image_by_id, update_image_status_and_result, ImageStatus
from app.services.s3_service import download_file_from_s3, upload_processed_file_to_s3

logger = get_task_logger(__name__)

celery_app = Celery("tasks")
celery_app.conf.broker_url = config.BROKER_URL
celery_app.conf.result_backend = config.RESULT_BACKEND

@celery_app.task
def process_image_task(image_id: int):
    
    db = SessionLocal()
    try:
        image_obj = get_image_by_id(db, image_id)
        if not image_obj:
            logger.error(f"Не найдено изображение с id={image_id}")
            return

        update_image_status_and_result(db, image_id, ImageStatus.PENDING)

        original_bytes = download_file_from_s3(image_obj.s3_path)
        processed_bytes = _process_image(
            original_bytes, 
            grain=image_obj.grain, 
            sharpness=image_obj.sharpness
        )

        processed_key = upload_processed_file_to_s3(processed_bytes)
        update_image_status_and_result(db, image_id, ImageStatus.SUCCESS, processed_s3_path=processed_key)

    except Exception as e:
        logger.exception(f"Ошибка при обработке изображения id={image_id}")
        update_image_status_and_result(db, image_id, ImageStatus.FAILED)
    finally:
        db.close()

def _process_image(image_bytes: bytes, grain: int, sharpness: int) -> bytes:
    
    with Image.open(io.BytesIO(image_bytes)) as img:
        inverted = ImageOps.invert(img.convert("RGB"))

        enhancer = ImageEnhance.Brightness(inverted)
        factor = 1.0 + (grain - 50)/100.0  
        grain_applied = enhancer.enhance(factor)


        sharp_enhancer = ImageEnhance.Sharpness(grain_applied)
        sharp_factor = 1.0 + (sharpness - 50)/50.0
        final_img = sharp_enhancer.enhance(sharp_factor)

        output = io.BytesIO()
        final_img.save(output, format="PNG")
        return output.getvalue()
