import os
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFace
from paths import CODE_DIR
from utils.config_utils import load_config


load_dotenv()

role = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
region = os.getenv("REGION")
hf_token = os.getenv("HF_TOKEN")


cfg = load_config()
# S3 paths
output_path = f"s3://{cfg['bucket']}/{cfg['output_path']}/"


# HuggingFace estimator configuration
huggingface_estimator = HuggingFace(
    entry_point="train_qlora.py",
    source_dir=CODE_DIR,
    instance_type=cfg["sagemaker_instance_type"],
    instance_count=1,
    role=role,
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    environment={
        "HF_MODEL_ID": "meta-llama/Llama-3.2-1B-Instruct",
        "HF_TOKEN": hf_token,
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
        "HF_DATASETS_CACHE": "/tmp/datasets_cache",
    },
    base_job_name=cfg["sagemaker_base_job_name"],
    output_path=output_path,
)

huggingface_estimator.fit()
