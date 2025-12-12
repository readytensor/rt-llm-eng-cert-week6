"""
bedrock_evaluate_batch.py

Download batch inference results from S3 and evaluate predictions.
Can evaluate results from pretrained or fine-tuned models.

Usage:
    python bedrock_evaluate_batch.py --pretrained <job_name>
    python bedrock_evaluate_batch.py --finetuned <job_name>
"""

import json
import os
import sys
import boto3
import yaml
from utils.data_utils import load_and_prepare_dataset
from utils.inference_utils import compute_rouge
from paths import CONFIG_FILE_PATH, BATCH_RESULTS_DIR


def download_results_from_s3(s3_client, bucket, prefix, local_dir):
    """
    Download all result files from S3 output location.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
        local_dir: Local directory to save files

    Returns:
        List of downloaded file paths
    """
    print(f"\nDownloading results from s3://{bucket}/{prefix}...")

    os.makedirs(local_dir, exist_ok=True)

    # List all objects in prefix
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        print("✗ No results found")
        return []

    downloaded_files = []

    for obj in response["Contents"]:
        s3_key = obj["Key"]

        # Skip directories
        if s3_key.endswith("/"):
            continue

        # Download file
        filename = os.path.basename(s3_key)
        local_path = os.path.join(local_dir, filename)

        s3_client.download_file(bucket, s3_key, local_path)
        downloaded_files.append(local_path)
        print(f"  ✓ {filename}")

    return downloaded_files


def parse_batch_results(result_files):
    """
    Parse downloaded JSONL result files.

    Args:
        result_files: List of result file paths

    Returns:
        List of prediction dictionaries
    """
    all_results = []

    for file_path in result_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                all_results.append(result)

    print(f"\n✓ Parsed {len(all_results)} predictions")
    return all_results


def match_with_references(predictions, val_dataset, field_map):
    """
    Match predictions with reference summaries using recordId.
    Bedrock batch jobs include recordId (1-indexed) to maintain order.
    
    Args:
        predictions: List of prediction dicts from Bedrock
        val_dataset: Original validation dataset
        field_map: Field mapping from config
    
    Returns:
        List of dicts with dialogue, reference, and prediction
    """
    # Create a dictionary mapping recordId to prediction
    pred_dict = {}
    for pred in predictions:
        record_id = int(pred.get('recordId', -1))
        generated = pred.get('modelOutput', {}).get('generation', '')
        pred_dict[record_id] = generated.strip()
    
    # Match predictions with dataset
    # recordId is 1-indexed, dataset is 0-indexed
    results = []
    
    for i, sample in enumerate(val_dataset):
        # recordId = i + 1 (convert 0-indexed to 1-indexed)
        prediction = pred_dict.get(i + 1, '')
        
        results.append({
            'dialogue': sample[field_map['input']],
            'reference': sample[field_map['output']],
            'prediction': prediction
        })
    
    return results


def run_evaluation(job_id:str):
    """
    Run evaluation on batch inference results.
    """
    # Load configuration
    print("Loading configuration...")
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Get config values
    model_id = cfg.get("bedrock_model_id", "model")
    bucket = cfg["bedrock_bucket"]
    region = os.getenv("AWS_REGION", "us-east-1")

    # Create S3 client
    s3_client = boto3.client("s3", region_name=region)

    print("\n" + "=" * 80)
    print("EVALUATE BATCH INFERENCE RESULTS")
    print("=" * 80)
    print(f"Job: {job_id}")
    print("=" * 80)

    # Download results from specific job folder
    output_dir = cfg["bedrock_batch_outputs_dir"]
    job_prefix = f"{output_dir}/{job_id}"
    local_dir = os.path.join(BATCH_RESULTS_DIR, job_id)
    
    result_files = download_results_from_s3(
        s3_client=s3_client, bucket=bucket, prefix=job_prefix, local_dir=local_dir
    )

    if not result_files:
        print(f"\n✗ No results found for job: {job_id}")
        print(f"   Looked in: s3://{bucket}/{job_prefix}")
        print("\n   Make sure the job completed successfully")
        return

    # Parse predictions
    predictions = parse_batch_results(result_files)

    # Load validation dataset for reference matching
    print("\nLoading validation dataset...")
    _, val_dataset, _ = load_and_prepare_dataset(cfg)
    field_map = cfg["dataset"]["field_map"]

    # Match with references
    print("\nMatching predictions with references...")
    final_results = match_with_references(predictions, val_dataset, field_map)

    # Save combined results
    output_file = os.path.join(local_dir, f"{job_id}_predictions.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for result in final_results:
            f.write(json.dumps(result) + "\n")

    print(f"✓ Results saved to: {output_file}")

    # Print sample results
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS")
    print("=" * 80)

    for i in range(min(3, len(final_results))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Dialogue: {final_results[i]['dialogue'][:100]}...")
        print(f"Reference: {final_results[i]['reference']}")
        print(f"Prediction: {final_results[i]['prediction']}")

    # Compute ROUGE metrics
    print("\n" + "=" * 80)
    print("COMPUTING ROUGE SCORES")
    print("=" * 80)

    # Extract predictions list
    preds = [r["prediction"] for r in final_results]

    # Compute ROUGE
    scores = compute_rouge(preds, val_dataset)

    # Display results
    results = {
        "job_name": job_id,
        "model_id": model_id,
        "num_samples": len(final_results),
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
    }

    print(f"\nJob: {results['job_name']}")
    print(f"Model ID: {results['model_id']}")
    print(f"Samples evaluated: {results['num_samples']}")
    print("\nROUGE Scores:")
    print(f"  ROUGE-1: {results['rouge1']:.4f}")
    print(f"  ROUGE-2: {results['rouge2']:.4f}")
    print(f"  ROUGE-L: {results['rougeL']:.4f}")

    # Save metrics
    metrics_file = os.path.join(local_dir, f"{job_id}_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Metrics saved to: {metrics_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Predictions: {output_file}")
    print(f"Metrics: {metrics_file}")
    print("="*80)

    return results

def main():
    # Check for job name argument
    if len(sys.argv) < 2:
        print("Usage: code/bedrock_evaluate_batch.py <job_id>")
        print("\nExample:")
        print("  code/bedrock_evaluate_batch.py j45wouwjfza7")
        return

    job_id = sys.argv[1]
    
    results = run_evaluation(job_id)
    return results





if __name__ == "__main__":
    main()