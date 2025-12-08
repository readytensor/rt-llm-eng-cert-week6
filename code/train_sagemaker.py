import os
from sagemaker import Session
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFace
from paths import CODE_DIR


load_dotenv()

role = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
bucket = os.getenv("BUCKET")
region = os.getenv("REGION")
hf_token = os.getenv("HF_TOKEN")


# Session and S3 paths
sess = Session()
output_path = f"s3://{bucket}/llama-3-2-1b-instruct/"


# HuggingFace estimator configuration
huggingface_estimator = HuggingFace(
    entry_point="train_qlora.py",
    source_dir=CODE_DIR,
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    sagemaker_session=sess,
    environment={
        "HF_MODEL_ID": "meta-llama/Llama-3.2-1B-Instruct",
        "HF_TOKEN": hf_token,
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
        "HF_DATASETS_CACHE": "/tmp/datasets_cache",
    },
    base_job_name="llama-3-2-1b-instruct-train",
    output_path=output_path,
)

huggingface_estimator.fit()
