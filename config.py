import os
from datetime import datetime

class Config:
    # Dataset
    DATASET = "cifar100"
    DATA_ROOT = "./data"
    NUM_CLASSES = 100

    # Models & modes
    MODELS = ["resnext50_32x4d", "densenet121"]
    TRAINING_MODES = ["scratch", "finetune"]

    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    EPOCHS_SCRATCH = 60
    EPOCHS_FINETUNE = 25

    LEARNING_RATE_SCRATCH = 0.1
    LEARNING_RATE_FINETUNE = 0.001
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9

    # LR milestones for scratch
    LR_MILESTONES = [30, 45]
    LR_GAMMA = 0.1

    DEVICE = "cuda"

    # Paths
    SAVE_DIR = "./checkpoints"
    LOG_DIR = "./runs"
    EXP_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")

    @classmethod
    def get_exp_name(cls, model_name, mode):
        return f"{model_name}_{mode}_{cls.EXP_NAME}"