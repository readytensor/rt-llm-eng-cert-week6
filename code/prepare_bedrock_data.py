import os
import json
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset
from paths import OUTPUTS_DIR
from create_s3_bucket import create_s3_bucket

load_dotenv()

# Load configuration
cfg = load_config()
region = os.getenv("REGION", "us-east-1")
bucket_name = cfg["bedrock_bucket"]

# Initialize S3 client
s3_client = boto3.client("s3", region_name=region)


def convert_to_bedrock_format(example, system_prompt=None):
    """
    Convert a samsum dataset example to Bedrock conversation format.

    Args:
        example: Dict with 'dialogue' and 'summary' keys
        system_prompt: Optional system prompt to include

    Returns:
        dict: Formatted for Bedrock fine-tuning
    """
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that creates concise summaries of conversations."

    # Create the bedrock conversation format
    bedrock_example = {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": system_prompt}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": f"Summarize the following conversation:\n\n{example['dialogue']}"
                    }
                ],
            },
            {"role": "assistant", "content": [{"text": example["summary"]}]},
        ],
    }

    return bedrock_example


def create_jsonl_file(dataset, output_file, system_prompt=None):
    """
    Convert dataset to JSONL format and save to file.

    Args:
        dataset: HuggingFace dataset
        output_file: Path to save JSONL file
        system_prompt: Optional system prompt
        max_samples: Optional limit on number of samples

    Returns:
        tuple: (output_file, number of examples)
    """
    print(f"Creating {output_file}...")
    num_examples = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            bedrock_example = convert_to_bedrock_format(example, system_prompt)
            f.write(json.dumps(bedrock_example) + "\n")
            num_examples += 1

    print(f"✓ Created {output_file} with {num_examples} examples")
    return output_file, num_examples


def upload_to_s3(local_file, s3_key):
    """
    Upload a file to S3.

    Args:
        local_file: Local file path
        s3_key: S3 key (path in bucket)

    Returns:
        str: S3 URI of the uploaded file
    """
    try:
        print(f"Uploading {local_file} to s3://{bucket_name}/{s3_key}...")
        s3_client.upload_file(local_file, bucket_name, s3_key)
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"✓ Uploaded to {s3_uri}")
        return s3_uri
    except ClientError as e:
        print(f"✗ Error uploading to S3: {e}")
        return None


def main():
    """
    Main function to prepare and upload Bedrock training data.
    """
    print("=" * 60)
    print("Prepare Bedrock Training Data from SamSum")
    print("=" * 60)

    # Create S3 bucket if it doesn't exist
    create_s3_bucket(bucket_name, region)

    # Configuration
    dataset_name = cfg["dataset"]["name"]
    system_prompt = cfg["task_instruction"]

    # Output configuration
    output_dir = os.path.join(OUTPUTS_DIR, cfg["bedrock_data_dir"])
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "validation.jsonl")

    # S3 paths
    train_s3_key = "llm-tuning-data/train.jsonl"
    val_s3_key = "llm-tuning-data/validation.jsonl"

    # Step 1: Load dataset
    print("\n" + "=" * 60)
    print("Step 1: Loading SamSum dataset from HuggingFace")
    print("=" * 60)

    try:
        print(f"Loading dataset: {dataset_name}")
        train, val, test = load_and_prepare_dataset(cfg)
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Train samples: {len(train)}")
        print(f"  - Validation samples: {len(val)}")
        print(f"  - Test samples: {len(test)}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Step 2: Create training JSONL
    print("\n" + "=" * 60)
    print("Step 2: Creating training JSONL file")
    print("=" * 60)

    train_file, num_train = create_jsonl_file(
        train,
        train_file,
        system_prompt=system_prompt,
    )

    # Step 3: Create validation JSONL
    print("\n" + "=" * 60)
    print("Step 3: Creating validation JSONL file")
    print("=" * 60)

    val_file, num_val = create_jsonl_file(
        val,
        val_file,
        system_prompt=system_prompt,
    )

    # Step 4: Validate the files
    print("\n" + "=" * 60)
    print("Step 4: Validating JSONL files")
    print("=" * 60)

    # Step 5: Upload to S3
    print("\n" + "=" * 60)
    print("Step 5: Uploading to S3")
    print("=" * 60)

    train_s3_uri = upload_to_s3(train_file, train_s3_key)
    val_s3_uri = upload_to_s3(val_file, val_s3_key)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Training data ready!")
    print(f"  - Local file: {train_file}")
    print(f"  - S3 URI: {train_s3_uri}")
    print(f"  - Examples: {num_train}")
    print()
    print(f"✓ Validation data ready!")
    print(f"  - Local file: {val_file}")
    print(f"  - S3 URI: {val_s3_uri}")
    print(f"  - Examples: {num_val}")


if __name__ == "__main__":
    main()
