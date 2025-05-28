import boto3
from botocore.exceptions import ClientError
import os

S3_INTERNAL_ENDPOINT = os.getenv("S3_INTERNAL_ENDPOINT", "http://minio:9000")
S3_EXTERNAL_ENDPOINT = os.getenv("S3_EXTERNAL_ENDPOINT", "http://localhost:9000")

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_INTERNAL_ENDPOINT,
    aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
    region_name="us-east-1"
)

BUCKET = "uploads"


def ensure_bucket():
    try:
        s3_client.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3_client.create_bucket(Bucket=BUCKET)


ensure_bucket()


def upload_to_s3(filename, content):
    s3_client.put_object(Bucket=BUCKET, Key=filename, Body=content)


def get_presigned_url(filename):
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": filename},
            ExpiresIn=3600
        )
        return url.replace(S3_INTERNAL_ENDPOINT, S3_EXTERNAL_ENDPOINT)
    except ClientError:
        return None
