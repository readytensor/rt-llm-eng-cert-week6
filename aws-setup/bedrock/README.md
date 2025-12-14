# AWS Setup Guide for Bedrock Examples

This guide walks through the IAM configuration needed to run the Bedrock inference examples.

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

Your IAM user needs permission to pass the Bedrock role to AWS services.

### 1.2 Create PassRole Policy and Attach to User Group

1. Go to IAM → Policies → Create policy
2. Click JSON tab
3. Copy contents from `aws-setup/bedrock/bedrock-passrole-policy.json`
4. **Replace `YOUR_ACCOUNT_ID`** with your AWS account ID
5. Policy name: `Bedrock-PassRole-Policy`
6. Create policy
7. Attach policy to user group `ai-engineer`:
   1. Go to IAM → User groups → `ai-engineer` -> Permissions -> Add permissions -> Attach Policy -> `Bedrock-PassRole-Policy`. You may need to filter the "by type" to "Customer Managed Policies" to locate the policy.

### 1.3 Create IAM User

(if not already created)

1. Go to IAM → Users → Create user
2. Username: Choose your username (e.g., `your-name`)
3. Add user to `ai-engineer` group
4. Create access key:
   - Go to user → Security credentials → Create access key
   - Choose "Command Line Interface (CLI)"
   - Download or save the Access Key ID and Secret Access Key

### 1.4 Configure AWS CLI

```bash
aws configure
```

Enter:

- Access Key ID
- Secret Access Key
- Default region: `us-east-1` (or your preferred region)
- Default output format: `json`

---

## Step 2: Create Bedrock Execution Role

This role allows Bedrock to access your S3 data and invoke models on your behalf.

### 2.1 Create Custom Policy

1. Go to IAM → Policies → Create policy
2. Click JSON tab
3. Copy contents from `aws-setup/bedrock/bedrock-execution-role-policy.json`
4. **Replace `YOUR_BUCKET_NAME`** with your actual bucket name (e.g., `bedrock-bucket-llm-eng-testing`)
5. Policy name: `Bedrock-Execution-Role-Policy` (you can name it anything, but make sure to replace it in the next step)
6. Create policy

### 2.2 Create IAM Role

1. Go to IAM → Roles → Create role
2. Trusted entity type: **Custom trust policy**
3. Copy contents from `aws-setup/bedrock/bedrock-trust-policy.json`
4. Click Next
5. Attach policy: `Bedrock-Execution-Role-Policy` (created above)
6. Role name: `Bedrock-Job-Execution-Role`
7. Create role

### 2.3 Save Role ARN

1. Go to IAM → Roles → `Bedrock-Job-Execution-Role`
2. Copy the ARN (e.g., `arn:aws:iam::123456789012:role/Bedrock-Job-Execution-Role`)
3. Add to `.env` file (create if not exists):

```bash
   BEDROCK_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/Bedrock-Job-Execution-Role
   AWS_REGION=us-east-1
```

---

## Step 3: Choose a LLM and Get Inference Profile ID

### 3.1 Choose a LLM

Go to AWS Console → Bedrock → Model Catalog. Choose a LLM.

Most models are already enabled. A few models like from Anthropic require approval. If required, request access on the model page.

### 3.2 Get Inference Profile ID

1. Go to Bedrock → Cross-region inference (left sidebar)
2. Click "Inference profiles" tab
3. Search for "Llama 3.2 1B Instruct"
4. Copy the Inference profile ID: `us.meta.llama3-2-1b-instruct-v1:0`
5. Add to `code/config.yaml`:

```yaml
bedrock_model_id: us.meta.llama3-2-1b-instruct-v1:0
```

---

## Verification Checklist

Before running the examples, verify:

- [ ] Your user-group `ai-engineer` created with `AmazonBedrockFullAccess` and `AmazonS3FullAccess` policies
- [ ] PassRole permission added to your user group `ai-engineer`
- [ ] IAM role `Bedrock-Job-Execution-Role` created
- [ ] Role ARN added to `.env` file
- [ ] Inference profile ID for chosen LLM added to `config.yaml`
- [ ] AWS CLI configured with credentials
- [ ] Dependencies installed in virtual environment

---

## Security Best Practices

For production use:

1. Use more restrictive IAM policies (principle of least privilege)
2. Enable S3 bucket encryption
3. Use VPC endpoints for Bedrock access
4. Rotate access keys regularly
5. Enable CloudTrail logging
