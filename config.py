"""
QuestCrafter Configuration
Global settings for data, model, and training pipeline
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Data settings
RAW_DATA_FILE = BASE_DIR / "train.csv"
VALIDATION_DATA_FILE = BASE_DIR / "validation.csv"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
VAL_JSONL = DATA_DIR / "val.jsonl"
TEST_JSONL = DATA_DIR / "test.jsonl"

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
MAX_SAMPLES = 5000  # Limit for efficient training

# Model settings
MODEL_NAME = "distilgpt2"  # Options: distilgpt2, gpt2, t5-small
PRETRAINED_MODEL_NAME = MODEL_NAME
FINETUNED_MODEL_DIR = MODELS_DIR / "quest_crafter_finetuned"

# Training settings
EPOCHS = 1  # Reduced for testing
BATCH_SIZE = 2  # Smaller batch for testing
LEARNING_RATE = 5e-5
WARMUP_STEPS = 5  # Reduced
MAX_LENGTH = 128  # Shorter sequences
SEED = 42

# Inference settings
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95

# Evaluation settings
NUM_TEST_PROMPTS = 50
EVALUATION_OUTPUT_FILE = OUTPUTS_DIR / "evaluation_results.json"
GENERATION_OUTPUT_FILE = OUTPUTS_DIR / "generations.jsonl"

# Demo settings
DEMO_PORT = 8501

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
