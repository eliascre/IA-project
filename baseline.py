"""
Baseline inference with pretrained model (no fine-tuning)
Tests model performance before training
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from config import (
    MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    SEED,
    OUTPUTS_DIR,
)

torch.manual_seed(SEED)


def load_model_and_tokenizer(model_name):
    """Load pretrained model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_quest(pipe, prompt, max_tokens=MAX_NEW_TOKENS):
    """Generate a quest from a prompt."""
    output = pipe(
        prompt,
        max_new_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        do_sample=True,
        num_return_sequences=1,
    )
    return output[0]['generated_text']


def create_test_prompts():
    """Create a set of diverse test prompts."""
    prompts = [
        "A brave knight must rescue a princess from a dark tower",
        "In a magical forest, a young mage discovers an ancient spell",
        "A thief in a bustling city discovers a hidden treasure map",
        "An adventurer wakes up in a mysterious dungeon",
        "A group of friends must stop an evil sorcerer",
        "In a desert city, a merchant finds a magical artifact",
        "A dragon awakens after centuries of sleep",
        "A lost sailor finds an island with strange creatures",
        "A young hero discovers they have magical powers",
        "In a cold mountain, warriors prepare for battle",
    ]
    return prompts


def run_baseline_inference(num_examples=10):
    """Run baseline inference on test prompts."""
    
    print("=" * 60)
    print("QuestCrafter Baseline Inference")
    print("=" * 60)
    
    # Load model
    print(f"\n1. Loading pretrained model: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    
    # Create test prompts
    test_prompts = create_test_prompts()[:num_examples]
    
    # Generate and store results
    results = []
    print(f"\n2. Generating {len(test_prompts)} quest samples...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n   [{i}/{len(test_prompts)}] Prompt: {prompt[:50]}...")
        
        generated_text = generate_quest(pipe, prompt)
        
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "model": MODEL_NAME,
            "stage": "baseline",
        })
        
        # Print first 150 chars of generation
        preview = generated_text[:150].replace('\n', ' ')
        print(f"   Output: {preview}...")
    
    # Save results
    output_file = OUTPUTS_DIR / "baseline_generations.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n3. Analysis & Common Issues")
    print(f"   - Check for repetition (same phrases)")
    print(f"   - Check coherence (logical flow)")
    print(f"   - Check prompt faithfulness (respects input)")
    print(f"   - Check length (not too short/long)")
    
    print("\n" + "=" * 60)
    print(f"✓ Baseline inference complete!")
    print(f"✓ Saved {len(results)} generations to {output_file}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_baseline_inference(num_examples=5)
