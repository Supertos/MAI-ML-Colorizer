from sqlalchemy import create_engine, Column, Integer, String, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum
from app import config

# формируем DSN для подключения к PostgreSQL
DATABASE_URL = (
    f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASS}"
    f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
)
print("DATABASE_URL=", repr(DATABASE_URL))

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ImageStatus(str, enum.Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

class ImageModel(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String(255))
    s3_path = Column(Text)
    processed_s3_path = Column(Text, nullable=True)
    status = Column(Enum(ImageStatus), default=ImageStatus.PENDING)
    grain = Column(Integer, default=50)
    sharpness = Column(Integer, default=50)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_image_record(db, original_filename, s3_path, grain, sharpness):
    new_image = ImageModel(
        original_filename=original_filename,
        s3_path=s3_path,
        status=ImageStatus.PENDING,
        grain=grain,
        sharpness=sharpness
    )
    db.add(new_image)
    db.commit()
    db.refresh(new_image)
    return new_image

def get_image_by_id(db, image_id: int):
    return db.query(ImageModel).filter(ImageModel.id == image_id).first()

def update_image_status_and_result(db, image_id: int, status: ImageStatus, processed_s3_path: str = None):
    image_obj = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if image_obj:
        image_obj.status = status
        if processed_s3_path:
            image_obj.processed_s3_path = processed_s3_path
        db.commit()
        db.refresh(image_obj)
    return image_obj
