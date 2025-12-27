# LLM Engineering & Deployment - Week 6 Code Examples

**Week 6: AWS Infrastructure for LLM Engineering**  
Part of the LLM Engineering & Deployment Certification Program

This repository contains examples for working with LLMs on AWS using two different approaches:

- **Amazon Bedrock** - Serverless inference with single and batch inference capabilities
- **Amazon SageMaker** - Managed training jobs with QLoRA fine-tuning, deployment, and evaluation

---

## Amazon Bedrock Examples

This section demonstrates how to perform inference using AWS Bedrock, including single inference requests and batch inference jobs with evaluation.

### 1. Environment Setup

Create a virtual environment:

```bash
python -m venv venv-bedrock
```

Activate the virtual environment:

```bash
# On Windows:
venv-bedrock\Scripts\activate

# On Mac/Linux:
source venv-bedrock/bin/activate
```

### 2. Dependency Installation

Install the required dependencies:

```bash
pip install -r requirements-bedrock.txt
```

### 3. IAM Setup

Before running the Bedrock examples, you need to configure IAM roles and permissions. Follow the detailed setup guide:

**[AWS Setup Guide for Bedrock](aws-setup/bedrock/README.md)**

This guide covers:

- Creating IAM users and groups
- Requesting Bedrock model access
- Creating Bedrock execution roles
- Configuring PassRole permissions

### 4. Environment Variables (.env file)

Create a `.env` file in the repository root directory with the following variables:

```bash
BEDROCK_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/Bedrock-Job-Execution-Role
AWS_REGION=us-east-1
```

Replace `YOUR_ACCOUNT_ID` with your AWS account ID and adjust the region if needed. The role ARN will be provided after completing the IAM setup steps above.

### 5. Update Configuration File

Update `code/config.yaml` with your Bedrock-specific configuration:

```yaml
bedrock_model_id: us.meta.llama3-2-1b-instruct-v1:0 # Inference profile ID
bedrock_bucket: your-unique-bucket-name # Your S3 bucket name
bedrock_data_dir: bedrock-samsum-dataset
bedrock_batch_outputs_dir: bedrock-batch-outputs
```

Make sure to replace `your-unique-bucket-name` with your actual S3 bucket name. The `bedrock_model_id` should match the inference profile ID from your region (see the IAM setup guide for details).

### 6. Running Bedrock Examples

Once the IAM setup is complete, you can run the following examples:

#### Single Inference

Run a single inference request on a Bedrock model:

```bash
cd code
python bedrock_inference_single.py
```

#### Batch Inference and Evaluation

Before running batch inference, prepare and upload the data to S3:

```bash
cd code
python prepare_bedrock_data.py
```

Then run batch inference on validation data and evaluate the results:

```bash
python bedrock_inference_batch.py
python bedrock_evaluate_batch.py
```

To monitor inference progress in the AWS Console: AWS Console → Bedrock → Batch inference (left menu)

---

## Amazon SageMaker Examples

This section demonstrates how to train LLMs using SageMaker with QLoRA fine-tuning, deploy models to endpoints, and perform inference and evaluation.

### 1. Environment Setup

Create a virtual environment:

```bash
python -m venv venv-sagemaker
```

Activate the virtual environment:

```bash
# On Windows:
venv-sagemaker\Scripts\activate

# On Mac/Linux:
source venv-sagemaker/bin/activate
```

### 2. Dependency Installation

Install the required dependencies:

```bash
pip install -r requirements-sagemaker.txt
```

### 3. IAM Setup

Before running the SageMaker examples, you need to configure IAM roles and permissions.

> **Note**: IAM setup documentation for SageMaker will be added to `aws-setup/sagemaker/README.md` soon.

### 4. Environment Variables (.env file)

Create a `.env` file in the repository root directory with the following variables:

```bash
SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
REGION=your_region
HF_TOKEN=your_huggingface_token
```

Replace `YOUR_ACCOUNT_ID` with your AWS account ID, adjust the region if needed, and add your Hugging Face token for accessing gated models. The role ARN will be provided after completing the IAM setup steps above.

### 5. Update Configuration File

Update `code/config.yaml` with your SageMaker-specific configuration:

```yaml
sagemaker_instance_type: ml.p3.2xlarge # Instance type for training
bucket: your-sagemaker-bucket-name # Your S3 bucket name
output_path: outputs/llama3-2-1b-samsum-finetune # S3 path for model outputs
sagemaker_base_job_name: llama-3-2-1b-instruct-train
```

Make sure to replace `your-sagemaker-bucket-name` with your actual S3 bucket name. You can adjust the instance type and other training parameters as needed.

### 6. Running SageMaker Examples

Once the IAM setup is complete, you can run the following:

#### Training

Launch a SageMaker training job with QLoRA:

```bash
cd code
python train_sagemaker.py
```

#### Deployment, Inference, and Evaluation

Use the Jupyter notebook to deploy your trained model and run inference and evaluation:

```bash
cd code
jupyter notebook deploy-llm-sagemaker.ipynb
```

IMPORTANT: Make sure to delete the endpoint after you are done with the exercise.

---

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**

- Share and adapt this material for non-commercial purposes
- Must give appropriate credit and indicate changes made
- Must distribute adaptations under the same license

See [LICENSE](LICENSE) for full terms.

---

## Contact

For questions or issues related to this repository, please refer to the course materials or contact your instructor.
