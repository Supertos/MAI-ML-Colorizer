import uuid
import boto3
from botocore.exceptions import ClientError
from app import config


s3_client = boto3.client(
    "s3",
    endpoint_url=config.S3_ENDPOINT,
    aws_access_key_id=config.S3_ACCESS_KEY,
    aws_secret_access_key=config.S3_SECRET_KEY
)

def upload_file_to_s3(file_bytes: bytes, original_filename: str) -> str:
    
    unique_id = uuid.uuid4().hex
    extension = original_filename.split('.')[-1]
    s3_key = f"original/{unique_id}.{extension}"

    s3_client.put_object(
        Bucket=config.S3_BUCKET_NAME,
        Key=s3_key,
        Body=file_bytes
    )
    return s3_key

def download_file_from_s3(s3_key: str) -> bytes:
    
    response = s3_client.get_object(Bucket=config.S3_BUCKET_NAME, Key=s3_key)
    return response['Body'].read()

def upload_processed_file_to_s3(file_bytes: bytes) -> str:
    
    unique_id = uuid.uuid4().hex
    s3_key = f"processed/{unique_id}.png"
    s3_client.put_object(
        Bucket=config.S3_BUCKET_NAME,
        Key=s3_key,
        Body=file_bytes
    )
    return s3_key

def get_s3_file_url(s3_key: str) -> str:
    
    # тут делаеи прямую ссылку на файл 
    
    return f"{config.S3_ENDPOINT}/{config.S3_BUCKET_NAME}/{s3_key}"
