import os

# Detect if running in SageMaker
IS_SAGEMAKER = os.path.exists("/opt/ml")

if IS_SAGEMAKER:
    # SageMaker paths
    ROOT_DIR = "/opt/ml"
    CODE_DIR = "/opt/ml/code"
    CONFIG_FILE_PATH = "/opt/ml/code/config.yaml"
    DATA_DIR = "/opt/ml/input/data"
    DATASETS_DIR = "/tmp/datasets"
    OUTPUTS_DIR = "/opt/ml/model"
    MODEL_DIR = "/opt/ml/model"
    WANDB_DIR = "/tmp/wandb"
else:
    # Local development paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CODE_DIR = os.path.join(ROOT_DIR, "code")
    CONFIG_FILE_PATH = os.path.join(CODE_DIR, "config.yaml")
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
    OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
    MODEL_DIR = os.path.join(DATA_DIR, "model")
    WANDB_DIR = os.path.join(DATA_DIR, "wandb")

BASELINE_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "baseline")
LORA_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "lora")


# Bedrock-specific paths
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
BATCH_RESULTS_DIR = os.path.join(DATA_DIR, "batch_results")
