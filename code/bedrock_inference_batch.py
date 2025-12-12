"""
bedrock_inference_batch.py

Submit a Bedrock batch inference job on validation data.
Monitors job completion and optionally runs evaluation.

Usage:
    python bedrock_inference_batch.py
"""

import os
import time
import boto3
import yaml
from dotenv import load_dotenv
from datetime import datetime
from paths import CONFIG_FILE_PATH
from bedrock_evaluate_batch import run_evaluation


load_dotenv()


def create_batch_inference_job(
    bedrock_client, model_id, input_s3_uri, output_s3_uri, job_name, role_arn
):
    """
    Create a Bedrock batch inference job.

    Args:
        bedrock_client: Bedrock client
        model_id: Model identifier
        input_s3_uri: S3 URI of input JSONL file
        output_s3_uri: S3 URI prefix for outputs
        job_name: Unique job name
        role_arn: IAM role ARN for Bedrock

    Returns:
        Tuple of (job_arn, job_id)
    """
    print("\nCreating batch inference job...")
    print(f"  Model: {model_id}")
    print(f"  Input: {input_s3_uri}")
    print(f"  Output: {output_s3_uri}")

    response = bedrock_client.create_model_invocation_job(
        jobName=job_name,
        modelId=model_id,
        inputDataConfig={"s3InputDataConfig": {"s3Uri": input_s3_uri}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
        roleArn=role_arn,
    )

    job_arn = response["jobArn"]
    # Extract job ID from ARN (last part after final slash)
    # ARN format: arn:aws:bedrock:region:account:model-invocation-job/j45wouwjfza7
    job_id = job_arn.split("/")[-1]
    
    print(f"\n✓ Job created: {job_arn}")
    print(f"✓ Job ID: {job_id}")
    return job_arn, job_id


def wait_for_job_completion(bedrock_client, job_arn, check_interval=60):
    """
    Poll job status until complete or failed.

    Args:
        bedrock_client: Bedrock client
        job_arn: Job ARN to monitor
        check_interval: Seconds between status checks

    Returns:
        Tuple of (success: bool, output_s3_uri: str)
    """
    print("\nMonitoring job status (this may take several minutes)...")

    while True:
        response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        status = response["status"]

        print(f"  Status: {status}")

        if status == "Completed":
            print("\n✓ Job completed successfully")
            output_uri = response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
            return True, output_uri
        elif status in ["Failed", "Stopped"]:
            print(f"\n✗ Job {status.lower()}")
            if "failureMessage" in response:
                print(f"  Error: {response['failureMessage']}")
            return False, None

        # Still running, wait and check again
        time.sleep(check_interval)


def run_batch_inference():
    # Load configuration
    print("Loading configuration...")
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Get config values
    model_id = cfg["bedrock_model_id"]
    bucket = cfg["bedrock_bucket"]
    data_dir = cfg["bedrock_data_dir"]
    output_dir = cfg["bedrock_batch_outputs_dir"]
    region = os.getenv("AWS_REGION", "us-east-1")
    role_arn = os.getenv("BEDROCK_ROLE_ARN")

    if not role_arn:
        print("✗ Error: BEDROCK_ROLE_ARN environment variable not set")
        print("  Please set it to your Bedrock IAM role ARN")
        return

    # S3 URIs
    input_s3_uri = f"s3://{bucket}/{data_dir}/validation.jsonl"
    output_s3_uri = f"s3://{bucket}/{output_dir}/"

    # Create Bedrock client
    bedrock_client = boto3.client("bedrock", region_name=region)

    # Create unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"batch-inference-{timestamp}"

    print("\n" + "=" * 80)
    print("BEDROCK BATCH INFERENCE JOB")
    print("=" * 80)
    print(f"Job name: {job_name}")
    print("=" * 80)

    # Create batch job and get job ID
    job_arn, job_id = create_batch_inference_job(
        bedrock_client=bedrock_client,
        model_id=model_id,
        input_s3_uri=input_s3_uri,
        output_s3_uri=output_s3_uri,
        job_name=job_name,
        role_arn=role_arn,
    )

    # Wait for completion
    success, output_uri = wait_for_job_completion(bedrock_client, job_arn)

    if not success:
        print("\n✗ Batch job failed")
        return

    print("\n" + "=" * 80)
    print("JOB COMPLETE")
    print("=" * 80)
    print(f"Job ARN: {job_arn}")
    print(f"Job ID: {job_id}")
    print(f"Job name: {job_name}")
    print(f"Output location: {output_uri}")
    
    # Ask if user wants to run evaluation immediately
    print("\n" + "=" * 80)
    print("Run evaluation now?")
    response = input("Enter 'y' to evaluate immediately, or 'n' to skip: ").lower().strip()
    
    if response == 'y':
        print("\n" + "=" * 80)
        print("RUNNING EVALUATION")
        print("=" * 80)
        
        # Pass the job_id to evaluation
        results = run_evaluation(job_id)
        
        if results:
            print("\n✓ Evaluation complete")
    else:
        print("\nTo evaluate later, run:")
        print(f"  python bedrock_evaluate_batch.py {job_id}")
        print("=" * 80)


if __name__ == "__main__":
    run_batch_inference()