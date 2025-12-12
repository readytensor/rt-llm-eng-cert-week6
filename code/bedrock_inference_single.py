"""
bedrock_inference_single.py

Run a single inference request on a Bedrock model using one validation sample.
Demonstrates basic Bedrock API usage for text generation.
"""

import json
import boto3
import yaml
from utils.data_utils import load_and_prepare_dataset, build_bedrock_llama_prompt
from paths import CONFIG_FILE_PATH


def invoke_bedrock_model(
    model_id, prompt, region="us-east-1", max_tokens=512, temperature=0.7
):
    """
    Call Bedrock model for text generation.

    Args:
        model_id: Bedrock model identifier (e.g., 'meta.llama3-2-1b-instruct-v1:0')
        prompt: Input text prompt
        region: AWS region
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text string
    """

    # Create Bedrock Runtime client
    client = boto3.client("bedrock-runtime", region_name=region)

    # Format request body for Llama models
    body = json.dumps(
        {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
    )

    # Invoke model
    response = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    # Parse response
    response_body = json.loads(response["body"].read())
    generated_text = response_body.get("generation", "")

    return generated_text


def main():
    # Load configuration
    print("Loading configuration...")
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load validation dataset
    print("\nLoading validation dataset...")
    _, val_dataset, _ = load_and_prepare_dataset(cfg)

    # Get one sample
    sample = val_dataset[0]

    # Build prompt
    task_instruction = cfg["task_instruction"]
    field_map = cfg["dataset"]["field_map"]

    dialogue = sample[field_map["input"]]
    reference_summary = sample[field_map["output"]]

    prompt = build_bedrock_llama_prompt(dialogue, task_instruction)

    # Run inference
    print("\n" + "=" * 80)
    print("RUNNING BEDROCK INFERENCE")
    print("=" * 80)
    print(f"\nModel: {cfg['bedrock_model_id']}")
    print(f"\nDialogue:\n{dialogue}")
    print(f"\nReference Summary:\n{reference_summary}")
    print("\nGenerating summary...")

    generated_summary = invoke_bedrock_model(
        model_id=cfg["bedrock_model_id"],
        prompt=prompt,
        max_tokens=128,
        temperature=0.7,
    )

    print(f"\nGenerated Summary:\n{generated_summary}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
