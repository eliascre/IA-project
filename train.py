"""
Fine-tuning script for QuestCrafter
Trains a small language model on instruction-response pairs
"""

import json
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from config import (
    MODEL_NAME,
    TRAIN_JSONL,
    VAL_JSONL,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WARMUP_STEPS,
    MAX_LENGTH,
    SEED,
    FINETUNED_MODEL_DIR,
    OUTPUTS_DIR,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_instruction_dataset(jsonl_file):
    """Load instruction-response dataset from JSONL."""
    data = {"instruction": [], "input": [], "output": []}
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data["instruction"].append(item.get("instruction", ""))
            data["input"].append(item.get("input", ""))
            data["output"].append(item.get("output", ""))
    
    return data


def format_instruction_prompt(instruction, input_text, output):
    """Format instruction into a prompt."""
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def preprocess_function(examples, tokenizer, max_length):
    """Tokenize and format examples."""
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        prompt = format_instruction_prompt(instruction, input_text, output)
        texts.append(prompt)
    
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    encoding["labels"] = encoding["input_ids"].clone()
    return encoding


def fast_train():
    """Fast training function with optimized settings for speed."""
    print("=" * 60)
    print("QuestCrafter Fast Training Mode")
    print("=" * 60)
    
    # Import transformers here
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
    except ImportError as e:
        print(f"✗ Error importing transformers: {e}")
        return
    
    # Check files exist
    if not TRAIN_JSONL.exists() or not VAL_JSONL.exists():
        print("✗ Error: Data files not found!")
        print("  Run preprocess.py first")
        return
    
    # Load tokenizer and model
    print(f"\n1. Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print(f"\n2. Loading datasets...")
    train_data = load_instruction_dataset(TRAIN_JSONL)
    val_data = load_instruction_dataset(VAL_JSONL)
    
    # Create HF datasets
    from datasets import Dataset
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Tokenize with smaller batch for speed
    print(f"\n3. Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=64,  # Larger batch for tokenization speed
        remove_columns=["instruction", "input", "output"],
    )
    
    val_tokenized = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=64,
        remove_columns=["instruction", "input", "output"],
    )
    
    # Fast training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUTS_DIR / "training_logs"),
        per_device_train_batch_size=2,  # Smaller batch size for testing
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        max_steps=150,  # Limit to 150 steps for testing
        learning_rate=1e-4,  # Higher learning rate for faster convergence
        warmup_steps=2,  # Fewer warmup steps
        weight_decay=0.01,
        logging_steps=10,  # More frequent logging for testing
        eval_strategy="no",  # Skip evaluation during training for speed
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=False,  # Skip loading best model
        seed=SEED,
        fp16=False,  # Disable FP16 for compatibility
        dataloader_num_workers=0,  # Disable parallel loading for testing
        dataloader_pin_memory=False,  # Disable pin memory for testing
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    print(f"\n4. Fast training model...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=None,  # No eval for speed
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save model
    print(f"\n5. Saving fast-trained model...")
    FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINETUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINETUNED_MODEL_DIR)
    
    print("\n" + "=" * 60)
    print(f"✓ Fast training complete!")
    print(f"✓ Model saved to {FINETUNED_MODEL_DIR}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("QuestCrafter Fine-tuning")
    print("=" * 60)
    
    # Import transformers here
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
    except ImportError as e:
        print(f"✗ Error importing transformers: {e}")
        return
    
    # Check files exist
    if not TRAIN_JSONL.exists() or not VAL_JSONL.exists():
        print("✗ Error: Data files not found!")
        print("  Run preprocess.py first")
        return
    
    # Load tokenizer and model
    print(f"\n1. Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print(f"\n2. Loading datasets...")
    train_data = load_instruction_dataset(TRAIN_JSONL)
    val_data = load_instruction_dataset(VAL_JSONL)
    
    # Create HF datasets
    from datasets import Dataset
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Tokenize
    print(f"\n3. Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=32,
        remove_columns=["instruction", "input", "output"],
    )
    
    val_tokenized = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=32,
        remove_columns=["instruction", "input", "output"],
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUTS_DIR / "training_logs"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=SEED,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    print(f"\n4. Training model...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save model
    print(f"\n5. Saving fine-tuned model...")
    FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINETUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINETUNED_MODEL_DIR)
    
    print("\n" + "=" * 60)
    print(f"✓ Fine-tuning complete!")
    print(f"✓ Model saved to {FINETUNED_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--fast":
        fast_train()
    else:
        main()
