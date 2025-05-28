from sqlalchemy import Column, String, DateTime, Integer
from datetime import datetime
from .database import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    s3_key = Column(String, nullable=False)
    colorized_s3_key = Column(String, nullable=True)
    anonymous_id = Column(String, index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    grain = Column(Integer, nullable=True)
    sharpness = Column(Integer, nullable=True)
