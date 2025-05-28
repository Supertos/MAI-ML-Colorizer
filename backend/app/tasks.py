# backend/app/tasks.py

import os
import logging
from io import BytesIO

import numpy as np
import cv2
from celery import Celery
from PIL import Image

from .celery_app import celery
from .s3_utils import upload_to_s3, s3_client, BUCKET
from .models import Image as ImageModel
from .database import SessionLocal
from .model_class.colorizer import build_colorizer

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(BASE_DIR, "5_128_2048_checkpoint.weights.h5")

_colorizer = None
def get_colorizer():
    global _colorizer
    if _colorizer is None:
        logger.info("Loading colorizer weights from %s", WEIGHTS_PATH)
        _colorizer = build_colorizer()
        _colorizer.load_weights(WEIGHTS_PATH)
        logger.info("Colorizer loaded")
    return _colorizer

def preprocess(image: np.ndarray, target_size=(128,128)) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    L_small = cv2.resize(L, target_size, interpolation=cv2.INTER_CUBIC)
    return L_small.reshape((1, target_size[0], target_size[1], 1))

def postprocess(L_orig: np.ndarray, ab_pred: np.ndarray) -> np.ndarray:
    h, w = L_orig.shape
    ab = ab_pred[0] * 128.0       # из [-1,1] → [-128,128]
    ab += 128.0                   # → [0,256]
    ab = np.clip(ab, 0, 255).astype(np.uint8)
    ab_large = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)

    lab = np.zeros((h, w, 3), dtype=np.uint8)
    lab[:, :, 0] = (L_orig * 255).astype(np.uint8)
    lab[:, :, 1:] = ab_large
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

@celery.task(bind=True)
def upload_file_task(self, s3_key: str, content: bytes):
    logger.info("Uploading original %s", s3_key)
    upload_to_s3(s3_key, content)

    output_key = f"inverted_{s3_key}"
    logger.info("Queueing colorize task for %s → %s", s3_key, output_key)
    invert_image_task.delay(s3_key, output_key)
    return f"Uploaded {s3_key}"

@celery.task(bind=True)
def invert_image_task(self, s3_key: str, output_key: str):
    try:
        logger.info("Colorizing %s → %s", s3_key, output_key)


        obj = s3_client.get_object(Bucket=BUCKET, Key=s3_key)
        img_bytes = obj['Body'].read()

   
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError("Failed to decode image")

   
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_orig = lab[:, :, 0] / 255.0

   
        model = get_colorizer()
        inp = preprocess(img_bgr)
        ab_pred = model.predict(inp)

       
        colorized = postprocess(L_orig, ab_pred)

     
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
        img_out = buf.tobytes()

     
        upload_to_s3(output_key, img_out)
        logger.info("Uploaded colorized to S3 as %s", output_key)

      
        db = SessionLocal()
        rec = db.query(ImageModel).filter(ImageModel.s3_key == s3_key).first()
        if rec:
            rec.colorized_s3_key = output_key
            db.commit()
            logger.info("DB updated for %s", s3_key)
        db.close()

        return f"Colorized and DB-updated: {output_key}"

    except Exception as e:
        logger.exception("Error in invert_image_task")
        raise
