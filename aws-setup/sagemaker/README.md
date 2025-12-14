# AWS Setup Guide for SageMaker Examples

This guide walks through the IAM configuration needed to run the SageMaker inference and fine-tuning examples.

## Prerequisites

- AWS Account with admin access (for initial setup)
- AWS CLI installed: `pip install awscli`
- Python 3.8+ installed

---

## Step 1: Create IAM User and Group

### 1.1 Create User Group (If not already created)

1. Go to AWS Console → IAM → User groups → Create group
2. Group name: `ai-engineer`
3. Attach policies:

   - `AmazonSageMakerFullAccess` (for SageMaker access)
   - `AmazonS3FullAccess` (for data upload/download)

   > **Note**: These are broad permissions for learning. In production, use more restrictive policies.

### 1.2 Create PassRole Policy and Attach to User Group

We need to attach a policy to your IAM user that allows reading IAM roles and passing role to SageMaker.

1. Goto IAM -> Policies -> Create policy
2. Click JSON tab
3. Copy contents from `aws-setup/sagemaker/sagemaker-passrole-policy.json`
4. Policy name: `SageMaker-PassRole-Policy`
5. Create policy
6. Attach policy to user group `ai-engineer`

### 1.3 Create IAM User (If not already created)

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

## Step 2: Create SageMaker Execution Role

This role allows SageMaker to access your S3 data and invoke models on your behalf.

### 2.1 Create Custom Policy

1. Go to IAM → Policies → Create policy
2. Click JSON tab
3. Copy contents from `aws-setup/sagemaker/sagemaker-execution-role-policy.json`
4. **Replace `YOUR_BUCKET_NAME`** with your actual bucket name (e.g., `sagemaker-bucket-llm-eng-testing`)
5. Policy name: `SageMaker-Execution-Role-Policy` (you can name it anything, but make sure to replace it in the next step)
6. Create policy

### 2.2 Create IAM Role

1. Go to IAM → Roles → Create role
2. Trusted entity type: **Custom trust policy**
3. Copy contents from `aws-setup/sagemaker/sagemaker-trust-policy.json`
4. Click Next
5. Attach policy: `SageMaker-Execution-Role-Policy` (created above)
6. Role name: `SageMaker-Execution-Role`
7. Create role

### 2.3 Save Role ARN

1. Go to IAM → Roles → `SageMaker-Execution-Role`
2. Copy the ARN (e.g., `arn:aws:iam::123456789012:role/SageMaker-Execution-Role`)
3. Add to `.env` file:

```bash
   SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMaker-Execution-Role
   AWS_REGION=us-east-1
```

---

---

## Verification Checklist

Before running the examples, verify:

- [ ] AWS CLI configured with credentials
- [ ] IAM role `SageMaker-Execution-Role` created
- [ ] Role ARN added to `.env` file
- [ ] PassRole permission added to your user group
- [ ] Dependencies installed in virtual environment

---
