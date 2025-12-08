import aws_cdk as cdk
from sagemaker_stack import SageMakerTrainingStack
from dotenv import load_dotenv
from utils.config_utils import load_config

load_dotenv()

cfg = load_config()

app = cdk.App()

SageMakerTrainingStack(
    app,
    "SageMakerTrainingStack",
    bucket_name=cfg["bucket"],
)

app.synth()
