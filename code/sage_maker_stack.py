from aws_cdk import (
    Stack,
    Duration,
    aws_s3 as s3,
    aws_iam as iam,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct


class SageMakerTrainingStack(Stack):
    """
    CDK Stack for SageMaker LLM Training Infrastructure

    Creates:
    - S3 bucket for training outputs and model artifacts
    - SageMaker execution role with full access to SageMaker, S3, and Bedrock
    - IAM developer user with full access to SageMaker, S3, Bedrock, and CloudWatch
    - Outputs for easy integration with training scripts
    """

    def __init__(
        self, scope: Construct, construct_id: str, bucket_name: str, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create S3 bucket for training outputs
        training_bucket = s3.Bucket(
            self,
            "TrainingOutputBucket",
            bucket_name=bucket_name,
            versioned=True,  # Enable versioning for model artifacts
            encryption=s3.BucketEncryption.S3_MANAGED,  # Server-side encryption
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,  # Security best practice
            removal_policy=RemovalPolicy.RETAIN,  # Don't delete bucket on stack deletion
            auto_delete_objects=False,  # Prevent accidental data loss
            lifecycle_rules=[
                # Clean up incomplete multipart uploads after 7 days
                s3.LifecycleRule(
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                    enabled=True,
                ),
            ],
        )

        # Create SageMaker execution role
        sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name="SageMakerTrainingRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for SageMaker LLM training jobs",
        )

        # Grant the role access to the training bucket
        training_bucket.grant_read_write(sagemaker_role)

        # Outputs for easy reference
        CfnOutput(
            self,
            "SageMakerRoleARN",
            value=sagemaker_role.role_arn,
            description="SageMaker Execution Role ARN",
        )

        CfnOutput(
            self,
            "TrainingBucketName",
            value=training_bucket.bucket_name,
            description="S3 Bucket for training outputs",
        )

        CfnOutput(
            self,
            "Region",
            value=self.region,
            description="AWS Region",
        )
