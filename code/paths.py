import os

# Detect if running in SageMaker
IS_SAGEMAKER = os.path.exists("/opt/ml")

if IS_SAGEMAKER:
    # SageMaker paths
    ROOT_DIR = "/opt/ml"
    CODE_DIR = "/opt/ml/code"
    CONFIG_FILE_PATH = "/opt/ml/code/config.yaml"
    DATA_DIR = "/opt/ml/input/data"
    DATASETS_DIR = "/tmp/datasets"  # Use tmp for caching
    OUTPUTS_DIR = "/opt/ml/model"  # <-- SageMaker uploads this to S3 automatically
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
GRIDSEARCH_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "grid_search")

EXPERIMENTS_DIR = os.path.join(DATA_DIR, "experiments")
OPENAI_FILES_DIR = os.path.join(EXPERIMENTS_DIR, "openai_files")

# DATASET_FILE = os.path.join(ROOT_DIR, "dataset.jsonl")

# CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

# DEEP_SPEED_CONFIG_DIR = os.path.join(CODE_DIR, "lesson3", "deepspeed_config")

# DEEP_SPEED_ZERO1_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero1.json")

# DEEP_SPEED_ZERO2_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero2.json")

# DEEP_SPEED_ZERO3_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero3.json")
