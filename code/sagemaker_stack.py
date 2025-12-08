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

        # S3 bucket access
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3BucketAccess",
                actions=[
                    "s3:GetObject*",
                    "s3:GetBucket*",
                    "s3:List*",
                    "s3:DeleteObject*",
                    "s3:PutObject",
                    "s3:PutObjectLegalHold",
                    "s3:PutObjectRetention",
                    "s3:PutObjectTagging",
                    "s3:PutObjectVersionTagging",
                    "s3:Abort*",
                ],
                resources=[
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
            )
        )

        # ECR access (pull training containers)
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRAccess",
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:DescribeRepositories",
                    "ecr:ListImages",
                ],
                resources=["*"],
            )
        )

        # CloudWatch Logs access
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogsAccess",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                    "logs:GetLogEvents",
                ],
                resources=["arn:aws:logs:*:*:log-group:/aws/sagemaker/*"],
            )
        )

        # SageMaker access
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerAccess",
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:ListTrainingJobs",
                    "sagemaker:AddTags",
                    "sagemaker:CreateModel",
                    "sagemaker:DescribeModel",
                ],
                resources=["*"],
            )
        )

        # CloudWatch Metrics
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchMetrics",
                actions=[
                    "cloudwatch:PutMetricData",
                    "cloudwatch:GetMetricData",
                    "cloudwatch:GetMetricStatistics",
                ],
                resources=["*"],
            )
        )

        # EC2 Network access
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="EC2NetworkAccess",
                actions=[
                    "ec2:CreateNetworkInterface",
                    "ec2:CreateNetworkInterfacePermission",
                    "ec2:DeleteNetworkInterface",
                    "ec2:DeleteNetworkInterfacePermission",
                    "ec2:DescribeNetworkInterfaces",
                    "ec2:DescribeVpcs",
                    "ec2:DescribeDhcpOptions",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeSecurityGroups",
                ],
                resources=["*"],
            )
        )

        # PassRole to SageMaker
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                sid="PassRoleToSageMaker",
                actions=["iam:PassRole"],
                resources=["*"],
                conditions={
                    "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
                },
            )
        )

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
