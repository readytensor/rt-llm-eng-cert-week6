# AWS LLM Training & Deployment

**Week 7: AWS Infrastructure for LLM Engineering**  
Part of the LLM Engineering & Deployment Certification Program

A comprehensive toolkit for fine-tuning and deploying LLMs on AWS using SageMaker and Bedrock. Learn managed training workflows, S3 integration, and enterprise-grade ML infrastructure.

## Overview

This repository provides hands-on code for two complementary AWS approaches to LLM fine-tuning:

1. **Amazon SageMaker** - Managed training jobs with full control over containers, instances, and training scripts
2. **AWS Bedrock** - Serverless fine-tuning with zero infrastructure management

Both workflows fine-tune Llama 3.2 1B Instruct on the SAMSum conversation summarization dataset, demonstrating how enterprise ML teams operationalize LLM training.

## Quick Start

### Prerequisites

Before running any training jobs, ensure you have:

1. **AWS Account** with IAM user configured (see Week 7 Lesson 2)
2. **AWS Credentials** configured via `aws configure` using aws cli
3. **SageMaker Execution Role** with S3 and CloudWatch permissions
4. **Bedrock Role** (for Bedrock workflows) with model access enabled
5. **Hugging Face Token** for accessing gated models

### Setup and Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
```

Then, install the dependencies:

```bash
# for SageMaker lessons
pip install -r requirements-sagemaker.txt
```

or

```bash
# for Bedrock lessons
pip install -r requirements-bedrock.txt
```

### Environment Setup

Create a `.env` file in the repository root:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
REGION=us-west-2

# SageMaker
SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole

# Bedrock
BEDROCK_ROLE_ARN=arn:aws:iam::123456789012:role/BedrockFineTuningRole

# Hugging Face
HF_TOKEN=your_hf_token
```

### SageMaker Training Workflow

**Step 1: Configure AWS CLI**

Before creating a bucket, make sure you have the AWS CLI installed. If not, [install it here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Then, configure your AWS credentials by running:

```bash
aws configure
```

This will prompt you for your AWS Access Key, Secret Access Key, region, and output format.

**Step 2: Create S3 Bucket**

If you don't have an S3 bucket yet, run:

```bash
cd code
python create_s3_bucket.py
```

You may also create the S3 bucket through the AWS Console.

**Step 2: Launch Training Job**

```bash
python train_sagemaker.py
```

This submits a managed SageMaker training job that:

- Provisions a GPU instance (configured in `config.yaml`)
- Runs your QLoRA training script
- Saves model artifacts to S3 automatically
- Tears down the instance when complete

Monitor progress in CloudWatch logs or the SageMaker console.

**Step 3: (Optional) Deploy to Endpoint**

Use the `deploy-llm-sagemaker.ipynb` notebook to deploy your trained model to a real-time inference endpoint.

### Bedrock Training Workflow

**Step 1: Prepare Training Data**

```bash
cd code
python prepare_bedrock_data.py
```

This script:

- Loads the SAMSum dataset from HuggingFace
- Converts it to Bedrock's conversation format
- Uploads training/validation JSONL files to S3

**Step 2: Launch Fine-Tuning Job**

```bash
python train_bedrock.py
```

This creates a Bedrock customization job that:

- Uses managed infrastructure (no instance provisioning)
- Fine-tunes the base model on your data
- Saves the customized model to your Bedrock account
- Provides a simple API for inference

Monitor progress in the AWS Bedrock Console under "Custom models".

## Configuration

Edit `code/config.yaml` to customize:

- **Model**: Base model ID and tokenizer
- **Dataset**: Dataset name, splits, and field mappings
- **Training**: Epochs, learning rate, batch size, LoRA parameters
- **SageMaker**: Instance type, bucket name, output paths
- **Bedrock**: Model ID, bucket name, hyperparameters

## Project Structure

```
├── code/
│   ├── train_sagemaker.py          # SageMaker training job launcher
│   ├── train_qlora.py              # QLoRA training script (runs in SageMaker)
│   ├── train_bedrock.py            # Bedrock fine-tuning job launcher
│   ├── prepare_bedrock_data.py     # Data preparation for Bedrock
│   ├── create_s3_bucket.py         # S3 bucket creation helper
│   ├── deploy-llm-sagemaker.ipynb  # SageMaker endpoint deployment notebook
│   ├── config.yaml                 # Centralized configuration
│   ├── paths.py                    # Path management (local vs SageMaker)
│   ├── requirements.txt            # Python dependencies
│   └── utils/
│       ├── config_utils.py         # Configuration loading
│       ├── data_utils.py           # Dataset loading and preparation
│       ├── model_utils.py          # Model loading utilities
│       └── inference_utils.py      # Inference helpers
├── sagemaker-policy.json           # IAM policy template for SageMaker
└── README.md
```

## Key Concepts

### SageMaker Training Jobs

- **Managed Infrastructure**: AWS provisions GPU instances automatically
- **Container-Based**: Uses prebuilt deep learning containers with PyTorch, Transformers, CUDA
- **S3 Integration**: Training scripts save to `/opt/ml/model/`, SageMaker uploads to S3
- **Cost Control**: Instances shut down automatically when training completes
- **Full Control**: You write the training script, choose instance types, configure hyperparameters

### AWS Bedrock Customization

- **Serverless**: No instance management, no containers, no SSH
- **API-Driven**: Simple API calls to create and monitor jobs
- **Managed Models**: Customized models appear in your Bedrock account
- **Simple Inference**: Deploy and invoke via Bedrock API
- **Less Control**: Limited hyperparameter tuning, fixed training approach

### When to Use Which?

- **SageMaker**: When you need full control, custom training logic, or specific library versions
- **Bedrock**: When you want simplicity, faster iteration, or don't need custom training code

## Output Locations

**SageMaker:**

```
s3://your-bucket/llm-training/output/sagemaker-huggingface-YYYY-MM-DD-HH-MM-SS/model.tar.gz
```

**Bedrock:**

- Custom models appear in AWS Bedrock Console → Custom models
- Training data: `s3://your-bucket/llm-tuning-data/train.jsonl`
- Output models: `s3://your-bucket/bedrock-models/{job_name}/`

## Common Issues & Debugging

### SageMaker

- **S3 Permission Errors**: Verify your execution role has `s3:GetObject` and `s3:PutObject` permissions
- **Bucket Region Mismatch**: Ensure bucket and training job are in the same region
- **Job Stuck in "Starting"**: Check IAM role trust relationship and GPU capacity
- **Import Errors**: Verify container versions match your dependencies
- **Disk Full**: Increase `volume_size` parameter for large models
- **Model Too Large**: SageMaker endpoints have a 5GB model size limit

### Bedrock

- **Model Access Not Enabled**: Enable model access in Bedrock Console → Model access
- **Incorrect Model ID**: Verify the model ID format for your region (e.g., `meta.llama3-2-1b-instruct-v1:0`)
- **IAM Permissions**: Ensure Bedrock role has `bedrock:CreateModelCustomizationJob` permission
- **Data Format Errors**: Verify JSONL files match Bedrock conversation schema

## Cost Management

⚠️ **Important**: Monitor your AWS costs carefully

- **SageMaker**: Charges per hour while instance is running (even if training is slow)
- **Bedrock**: Charges per customization job and per inference token
- **S3**: Storage costs are minimal, but watch for large model artifacts
- **CloudWatch Logs**: Can accumulate costs if not managed

**Best Practices:**

- Always delete SageMaker endpoints when done testing
- Use smaller instance types for experimentation (`ml.g5.xlarge` vs `ml.p5.48xlarge`)
- Set up billing alerts in AWS Cost Management
- Monitor CloudWatch for unexpected resource usage

## Requirements

- Python 3.8+
- AWS Account with appropriate IAM permissions
- CUDA-compatible GPU (for local testing only; SageMaker/Bedrock provide GPUs)
- ~10GB disk space for datasets and model artifacts

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Hugging Face SageMaker Integration](https://huggingface.co/docs/sagemaker/index)
- Week 7 Lesson Materials (SageMaker and Bedrock lessons)

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**

- Share and adapt this material for non-commercial purposes
- Must give appropriate credit and indicate changes made
- Must distribute adaptations under the same license

See [LICENSE](LICENSE) for full terms.
