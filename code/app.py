#!/usr/bin/env python3
import aws_cdk as cdk
import os
from sage_maker_stack import SageMakerTrainingStack
from dotenv import load_dotenv

load_dotenv()

bucket_name = os.getenv("BUCKET")
app = cdk.App()

SageMakerTrainingStack(
    app,
    "SageMakerTrainingStack",
    bucket_name=bucket_name,
)

app.synth()
