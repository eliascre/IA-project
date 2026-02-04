"""
Data preprocessing pipeline for QuestCrafter
Converts raw data to instruction/response format (JSONL)
"""

import csv
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import (
    RAW_DATA_FILE,
    VALIDATION_DATA_FILE,
    TRAIN_JSONL,
    VAL_JSONL,
    TEST_JSONL,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    MAX_SAMPLES,
    SEED,
)

random.seed(SEED)


def load_csv_data(csv_file):
    """Load CSV data and return list of records."""
    data = []
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if MAX_SAMPLES and i >= MAX_SAMPLES:
                break
            # Clean and validate
            if 'text' in row and row['text'].strip():
                data.append({'text': row['text'].strip()})
    return data


def create_instruction_format(stories):
    """
    Convert stories to instruction/response format.
    Format: {instruction, response} or {input, output}
    """
    formatted_data = []
    
    quest_templates = [
        "Create a fantasy quest from this story:",
        "Transform this into a quest description:",
        "Write a quest based on this narrative:",
        "Generate a quest prompt from:",
        "Adapt this into a game quest:",
    ]
    
    for story in stories:
        text = story['text']
        
        # Use random template for variety
        template = random.choice(quest_templates)
        
        # For TinyStories, we create instruction-response pairs
        # The instruction asks to create a quest from the story
        formatted_data.append({
            "instruction": template,
            "input": text[:100],  # First 100 chars as context
            "output": text,  # Full story as the "quest response"
        })
    
    return formatted_data


def save_jsonl(data, output_file):
    """Save list of dicts as JSONL."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(data)} records to {output_file}")


def split_and_save_data(all_data):
    """Split data into train/val/test and save."""
    
    # First split: 80% train, 20% temp (val+test)
    train_data, temp_data = train_test_split(
        all_data,
        train_size=TRAIN_SPLIT,
        random_state=SEED
    )
    
    # Second split: split temp into val and test (50-50 of 20%)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=0.5,
        random_state=SEED
    )
    
    # Save splits
    save_jsonl(train_data, TRAIN_JSONL)
    save_jsonl(val_data, VAL_JSONL)
    save_jsonl(test_data, TEST_JSONL)
    
    print(f"\nData split summary:")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(all_data)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/len(all_data)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/len(all_data)*100:.1f}%)")


def main():
    print("=" * 60)
    print("QuestCrafter Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Load raw data
    print("\n1. Loading raw data...")
    if RAW_DATA_FILE.exists():
        stories = load_csv_data(RAW_DATA_FILE)
        print(f"   ✓ Loaded {len(stories)} stories from {RAW_DATA_FILE.name}")
    else:
        print(f"   ✗ Error: {RAW_DATA_FILE} not found!")
        return
    
    # Create instruction format
    print("\n2. Creating instruction/response format...")
    formatted_data = create_instruction_format(stories)
    print(f"   ✓ Created {len(formatted_data)} instruction-response pairs")
    
    # Split and save
    print("\n3. Splitting into train/val/test...")
    split_and_save_data(formatted_data)
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
