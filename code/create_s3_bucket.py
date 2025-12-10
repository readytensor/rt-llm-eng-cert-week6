import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from utils.config_utils import load_config

load_dotenv()


cfg = load_config()
region = os.getenv("REGION")
bucket_name = cfg["bucket"]


def create_s3_bucket(bucket_name, region):
    """
    Create an S3 bucket with the specified name.

    Args:
        bucket_name (str): Name of the bucket to create
        region (str): AWS region where the bucket will be created

    Returns:
        bool: True if bucket was created, False otherwise
    """
    try:
        s3_client = boto3.client("s3", region_name=region)

        # For us-east-1, don't specify LocationConstraint
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )

        print(f"Successfully created bucket: {bucket_name}")
        return True

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "BucketAlreadyOwnedByYou":
            print(f"Bucket {bucket_name} already exists and is owned by you")
            return True
        elif error_code == "BucketAlreadyExists":
            print(f"Bucket {bucket_name} already exists and is owned by someone else")
            return False
        else:
            print(f"Error creating bucket: {e}")
            return False


if __name__ == "__main__":
    create_s3_bucket(bucket_name, region)
