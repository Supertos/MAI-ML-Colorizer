
from app.s3_utils import get_presigned_url
from fastapi import FastAPI, UploadFile, File, Depends, Form
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from app.database import get_db
from app.models import Image
from app.tasks import upload_file_task
from fastapi import HTTPException, status

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}
app = FastAPI()

DEFAULT_GRAIN = 10
DEFAULT_SHARPNESS = 5

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # или конкретный адрес фронта, например "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    grain: int = DEFAULT_GRAIN,
    sharpness: int = DEFAULT_SHARPNESS,
    anonymous_id: str = Form(None),  
    db: Session = Depends(get_db)
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": "invalid_file_type",
            "message": "Only PNG and JPEG images are allowed."
        }
    )

    content = await file.read()


    unique_filename = f"{uuid4()}_{file.filename}"
    s3_key = unique_filename

    if not anonymous_id:
        anonymous_id = str(uuid4())

    # Сохраняем в БД
    image = Image(
        filename=file.filename,
        s3_key=s3_key,
        anonymous_id=anonymous_id,
        grain=grain,
        sharpness=sharpness,
    )
    db.add(image)
    db.commit()
    db.refresh(image)

    upload_file_task.delay(s3_key, content)

    return {
        "image_id": image.id,
        "original_filename": file.filename,
        "s3_key": s3_key,
        "anonymous_id": anonymous_id
    }


@app.get("/images/{anonymous_id}")
def list_user_images(anonymous_id: str, db: Session = Depends(get_db)):
    images = db.query(Image).filter(Image.anonymous_id == anonymous_id).all()

    if not images:
        return {"message": "None"}

    return [
        {
            "id": image.id,
            "filename": image.filename,
            "s3_key": image.s3_key,
            "created_at": image.created_at,
            "download_url": get_presigned_url(image.s3_key),
            "inverted_download_url": get_presigned_url(image.colorized_s3_key) if image.colorized_s3_key else None
        }
        for image in images
    ]
