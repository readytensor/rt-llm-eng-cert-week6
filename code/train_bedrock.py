import os
import json
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from utils.config_utils import load_config

load_dotenv()

# Load configuration
cfg = load_config()
region = os.getenv("REGION", "us-east-1")
bucket_name = cfg["bedrock_bucket"]
role_arn = os.getenv("BEDROCK_ROLE_ARN")  # Get from environment variable

# Initialize boto3 client
bedrock_client = boto3.client("bedrock", region_name=region)


def create_fine_tuning_job(
    job_name,
    base_model_id,
    training_data_uri,
    validation_data_uri,
    output_data_uri,
    role_arn,
    hyperparameters,
):
    """
    Create a Bedrock fine-tuning job.

    Args:
        job_name: Unique name for the fine-tuning job
        base_model_id: ID of the base model
        training_data_uri: S3 URI of training data
        validation_data_uri: S3 URI of validation data
        output_data_uri: S3 URI for output model
        role_arn: ARN of IAM role for Bedrock
        hyperparameters: Dict of hyperparameters

    Returns:
        str: Job ARN if successful, None otherwise
    """
    try:
        print(f"Creating fine-tuning job: {job_name}")
        print(f"Base model: {base_model_id}")
        print(f"Training data: {training_data_uri}")
        print(f"Validation data: {validation_data_uri}")
        print(f"Output location: {output_data_uri}")
        print(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")

        config = {
            "jobName": job_name,
            "customModelName": f"{job_name}-model",
            "roleArn": role_arn,
            "baseModelIdentifier": base_model_id,
            "trainingDataConfig": {"s3Uri": training_data_uri},
            "outputDataConfig": {"s3Uri": output_data_uri},
            "hyperParameters": hyperparameters,
        }

        if validation_data_uri:
            config["validationDataConfig"] = {
                "validators": [{"s3Uri": validation_data_uri}]
            }

        response = bedrock_client.create_model_customization_job(**config)

        job_arn = response["jobArn"]
        print(f"\nFine-tuning job created successfully!")
        print(f"Job ARN: {job_arn}")
        print(f"\nMonitor progress in AWS Bedrock Console:")
        print(f"  Bedrock > Custom models > {job_name}")

        return job_arn

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        print(f"\nError creating fine-tuning job:")
        print(f"  Code: {error_code}")
        print(f"  Message: {error_message}")
        return None


def list_available_models():
    """List foundation models that support fine-tuning."""
    try:
        print("\nAvailable models for fine-tuning:")
        response = bedrock_client.list_foundation_models()
        models = [
            m
            for m in response.get("modelSummaries", [])
            if "FINE_TUNING" in m.get("customizationsSupported", [])
        ]

        for model in models:
            print(f"  {model['modelId']} - {model['modelName']}")
        print()
    except ClientError as e:
        print(f"Error listing models: {e}")


def main():
    """Launch a Bedrock fine-tuning job."""
    print("=" * 60)
    print("Bedrock Fine-Tuning - Llama 3.2 1B")
    print("=" * 60)

    list_available_models()

    if not role_arn:
        print("\nError: BEDROCK_ROLE_ARN environment variable not set")
        print("Set it in your .env file:")
        print("  BEDROCK_ROLE_ARN=arn:aws:iam::123456789012:role/BedrockFineTuningRole")
        return

    # Configuration
    job_name = f"llama32-1b-samsum-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    base_model_id = cfg.get("bedrock_model_id", "meta.llama3-2-1b-instruct-v1:0")

    # S3 URIs
    training_data_uri = f"s3://{bucket_name}/llm-tuning-data/train.jsonl"
    validation_data_uri = f"s3://{bucket_name}/llm-tuning-data/validation.jsonl"
    output_data_uri = f"s3://{bucket_name}/bedrock-models/{job_name}/"

    # Hyperparameters
    hyperparameters = {
        "epochCount": str(cfg.get("num_epochs", 1)),
        "batchSize": str(cfg.get("batch_size", 4)),
        "learningRate": str(cfg.get("learning_rate", 0.0002)),
    }

    # Create fine-tuning job
    print()
    job_arn = create_fine_tuning_job(
        job_name=job_name,
        base_model_id=base_model_id,
        training_data_uri=training_data_uri,
        validation_data_uri=validation_data_uri,
        output_data_uri=output_data_uri,
        role_arn=role_arn,
        hyperparameters=hyperparameters,
    )

    if not job_arn:
        print("\nFailed to create fine-tuning job")
        print("\nCommon issues:")
        print("  1. Model access not enabled in Bedrock console")
        print("  2. Incorrect base_model_id for your region")
        print("  3. IAM role lacks necessary permissions")
        return


if __name__ == "__main__":

    main()
