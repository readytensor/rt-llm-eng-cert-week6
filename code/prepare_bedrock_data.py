"""
Prepare and upload training data for Bedrock fine-tuning.
Formats dataset into JSONL with Llama's chat template and uploads to S3.
"""

import json
import os
import yaml
from utils.data_utils import load_and_prepare_dataset
from paths import CONFIG_FILE_PATH, DATA_DIR
from create_s3_bucket import create_s3_bucket


def format_sample_for_bedrock(sample, field_map, task_instruction):
    """
    Format a single sample into Bedrock's expected JSONL format for Llama models.

    Bedrock expects:
    - "prompt": User message with Llama chat template tags
    - "completion": Assistant response with closing tag

    Args:
        sample: Dataset sample with input/output fields
        field_map: Mapping of dataset fields (input -> dialogue, output -> summary)
        task_instruction: System/task instruction text

    Returns:
        Dictionary with "prompt" and "completion" keys
    """
    # Get input and output from sample
    dialogue = sample[field_map["input"]]
    summary = sample[field_map["output"]]

    # Build user message
    user_message = f"{task_instruction}\n\n## Dialogue:\n{dialogue}\n## Summary:"

    # Format with Llama chat template
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{user_message}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    # Completion should include closing tag
    completion = f" {summary}<|eot_id|>"

    return {"prompt": prompt, "completion": completion}


def save_dataset_as_jsonl(dataset, output_file, field_map, task_instruction):
    """
    Convert dataset to JSONL format and save locally.

    Args:
        dataset: HuggingFace dataset split
        output_file: Path to save JSONL file
        field_map: Field mapping from config
        task_instruction: Task instruction from config
    """
    print(f"Formatting {len(dataset)} samples...")

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in dataset:
            formatted = format_sample_for_bedrock(sample, field_map, task_instruction)
            f.write(json.dumps(formatted) + "\n")

    print(f"✓ Saved to: {output_file}")


def upload_to_s3(local_file, bucket, s3_key):
    """
    Upload file to S3 bucket.

    Args:
        local_file: Path to local file
        bucket: S3 bucket name
        s3_key: S3 object key (path within bucket)

    Returns:
        S3 URI of uploaded file
    """
    import boto3

    s3_client = boto3.client("s3")

    print(f"Uploading to s3://{bucket}/{s3_key}...")

    try:
        s3_client.upload_file(local_file, bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"
        print("✓ Upload complete")
        return s3_uri
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        raise


def main():
    # Load configuration
    print("Loading configuration...")
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Get config values
    bucket = cfg["bedrock_bucket"]
    data_dir = cfg["bedrock_data_dir"]
    region = os.getenv("REGION", "us-east-1")

    # Ensure bucket exists
    print("\nChecking S3 bucket...")
    if not create_s3_bucket(bucket, region):
        print("✗ Failed to create or access bucket. Exiting.")
        return

    # Load all dataset splits
    print("\nLoading dataset splits...")
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset(cfg)

    # Prepare output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Get dataset config
    field_map = cfg["dataset"]["field_map"]
    task_instruction = cfg["task_instruction"]

    # Process each split
    splits = {
        "training": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    }

    s3_uris = {}

    print("\n" + "=" * 80)
    print("PREPARING AND UPLOADING DATASET SPLITS")
    print("=" * 80)

    for split_name, dataset in splits.items():
        print(f"\n--- {split_name.upper()} SPLIT ---")

        # Save locally
        local_file = os.path.join(DATA_DIR, f"bedrock_{split_name}_data.jsonl")
        save_dataset_as_jsonl(dataset, local_file, field_map, task_instruction)

        # Upload to S3
        s3_key = f"{data_dir}/{split_name}.jsonl"
        s3_uri = upload_to_s3(local_file, bucket, s3_key)
        s3_uris[split_name] = s3_uri

    # Print summary
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print("\nS3 URIs:")
    print(f"  Training:   {s3_uris['training']}")
    print(f"  Validation: {s3_uris['validation']}")
    print(f"  Test:       {s3_uris['test']}")
    print("\nNext steps:")
    print("  • Use training URI for Bedrock fine-tuning job")
    print("  • Use validation URI for batch inference and evaluation")
    print("=" * 80)


if __name__ == "__main__":
    main()
