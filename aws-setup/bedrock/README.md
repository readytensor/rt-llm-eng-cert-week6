# AWS Setup Guide for Bedrock Examples

This guide walks through the IAM configuration needed to run the Bedrock inference and fine-tuning examples.

## Prerequisites

- AWS Account with admin access (for initial setup)
- AWS CLI installed: `pip install awscli`
- Python 3.8+ installed

---

## Step 1: Create IAM User and Group

### 1.1 Create User Group

1. Go to AWS Console → IAM → User groups → Create group
2. Group name: `ai-engineer`
3. Attach policies:

   - `AmazonBedrockFullAccess` (for model access and job management)
   - `AmazonS3FullAccess` (for data upload/download)

   > **Note**: These are broad permissions for learning. In production, use more restrictive policies.

### 1.2 Create IAM User

1. Go to IAM → Users → Create user
2. Username: Choose your username (e.g., `your-name`)
3. Add user to `ai-engineer` group
4. Create access key:
   - Go to user → Security credentials → Create access key
   - Choose "Command Line Interface (CLI)"
   - Download or save the Access Key ID and Secret Access Key

### 1.3 Configure AWS CLI

```bash
aws configure
```

Enter:

- Access Key ID
- Secret Access Key
- Default region: `us-east-1` (or your preferred region)
- Default output format: `json`

---

## Step 2: Request Bedrock Model Access

1. Go to AWS Console → Bedrock → Model access (left sidebar)
2. Click "Manage model access"
3. Find and enable:
   - **Llama 3.2 1B Instruct** (Meta)
4. Click "Save changes"
5. Wait for "Access granted" status (usually instant)

### Get Inference Profile ID

1. Go to Bedrock → Cross-region inference (left sidebar)
2. Click "Inference profiles" tab
3. Search for "Llama 3.2 1B Instruct"
4. Copy the Inference profile ID: `us.meta.llama3-2-1b-instruct-v1:0`
5. Add to `code/config.yaml`:

```yaml
bedrock_model_id: us.meta.llama3-2-1b-instruct-v1:0
```

---

## Step 3: Create Bedrock Execution Role

This role allows Bedrock to access your S3 data and invoke models on your behalf.

### 3.1 Create Custom Policy

1. Go to IAM → Policies → Create policy
2. Click JSON tab
3. Copy contents from `aws-setup/bedrock-job-execution-policy.json`
4. **Replace `YOUR_BUCKET_NAME`** with your actual bucket name (e.g., `bedrock-bucket-llm-eng-testing`)
5. Policy name: `Bedrock-Job-Execution-Policy`
6. Create policy

### 3.2 Create IAM Role

1. Go to IAM → Roles → Create role
2. Trusted entity type: **Custom trust policy**
3. Copy contents from `aws-setup/bedrock-role-trust-policy.json`
4. Click Next
5. Attach policy: `Bedrock-Job-Execution-Policy` (created above)
6. Role name: `Bedrock-Job-Execution-Role`
7. Create role

### 3.3 Save Role ARN

1. Go to IAM → Roles → `Bedrock-Job-Execution-Role`
2. Copy the ARN (e.g., `arn:aws:iam::123456789012:role/Bedrock-Job-Execution-Role`)
3. Add to `.env` file:

```bash
   BEDROCK_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/Bedrock-Job-Execution-Role
   AWS_REGION=us-east-1
```

---

## Step 4: Allow PassRole Permission

Your IAM user needs permission to pass the Bedrock role to AWS services.

### 4.1 Create PassRole Policy

1. Go to IAM → User groups → `ai-engineer`
2. Click "Add permissions" → "Create inline policy"
3. Click JSON tab
4. Copy contents from `aws-setup/bedrock-passrole-policy.json`
5. **Replace `YOUR_ACCOUNT_ID`** with your AWS account ID
6. Policy name: `Bedrock-PassRole-Policy`
7. Create policy

---

## Step 5: Update Configuration Files

### 5.1 Update config.yaml

Edit `code/config.yaml`:

```yaml
bedrock_model_id: us.meta.llama3-2-1b-instruct-v1:0
bedrock_bucket: your-unique-bucket-name # Must be globally unique
bedrock_data_dir: bedrock-samsum-dataset
bedrock_batch_outputs_pretrained: bedrock-batch-outputs-pretrained
bedrock_batch_outputs_finetuned: bedrock-batch-outputs-finetuned
```

### 5.2 Create .env file

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
BEDROCK_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/Bedrock-Job-Execution-Role
AWS_REGION=us-east-1
```

---

## Step 6: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements-bedrock.txt
```

---

## Verification Checklist

Before running the examples, verify:

- [ ] AWS CLI configured with credentials
- [ ] Llama 3.2 1B model access granted in Bedrock
- [ ] Inference profile ID added to `config.yaml`
- [ ] IAM role `Bedrock-Job-Execution-Role` created
- [ ] Role ARN added to `.env` file
- [ ] PassRole permission added to your user group
- [ ] Dependencies installed in virtual environment

---

## Running the Examples

### Example 1: Single Inference

```bash
python code/bedrock_inference_single.py
```

### Example 2: Batch Inference (Pretrained)

```bash
# Prepare and upload data to S3
python code/prepare_bedrock_data.py

# Run batch inference job
python code/bedrock_inference_batch.py --pretrained

# Evaluate results
python code/bedrock_evaluate_batch.py --pretrained
```

### Example 3: Fine-tuning

```bash
# Data should already be uploaded from Example 2

# Start fine-tuning job
python code/bedrock_finetune.py

# After training completes, run batch inference with fine-tuned model
python code/bedrock_inference_batch.py --finetuned

# Evaluate fine-tuned results
python code/bedrock_evaluate_batch.py --finetuned
```

---

## Monitoring Jobs

- **Batch Inference**: AWS Console → Bedrock → Batch inference (left menu)
- **Fine-tuning**: AWS Console → Bedrock → Custom models → Jobs tab

---

## Troubleshooting

### AccessDeniedException

- Check that model access is granted in Bedrock console
- Verify role ARN is correct in `.env`
- Ensure PassRole policy is attached to your user group

### S3 Permission Errors

- Update bucket name in `bedrock-job-execution-policy.json`
- Verify policy is attached to the Bedrock role

### "User is not authorized to perform: iam:PassRole"

- Attach `bedrock-passrole-policy.json` to your IAM user group

---

## Cost Considerations

- **Inference**: Charged per input/output token
- **Batch Inference**: Lower cost than real-time inference
- **Fine-tuning**: Charged per token processed during training
- **S3 Storage**: Minimal cost for small datasets

Always clean up resources after learning exercises to avoid unnecessary charges.

---

## Security Best Practices

For production use:

1. Use more restrictive IAM policies (principle of least privilege)
2. Enable S3 bucket encryption
3. Use VPC endpoints for Bedrock access
4. Rotate access keys regularly
5. Enable CloudTrail logging
