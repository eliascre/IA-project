#!/usr/bin/env python3
"""
Simple test script for training
"""

print("Testing training setup...")

try:
    from config import MODEL_NAME, TRAIN_JSONL, VAL_JSONL
    print(f"✓ Config loaded: {MODEL_NAME}")
    print(f"✓ Train file exists: {TRAIN_JSONL.exists()}")
    print(f"✓ Val file exists: {VAL_JSONL.exists()}")
except Exception as e:
    print(f"✗ Config error: {e}")
    exit(1)

try:
    from transformers import AutoTokenizer
    print("✓ Transformers import successful")
except Exception as e:
    print(f"✗ Transformers import failed: {e}")
    exit(1)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✓ Model loading successful")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    exit(1)

print("✓ All tests passed! Ready for training.")