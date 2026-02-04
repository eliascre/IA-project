"""
Evaluation script for QuestCrafter
Compares baseline vs fine-tuned models
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import (
    MODEL_NAME,
    FINETUNED_MODEL_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    SEED,
    OUTPUTS_DIR,
)

torch.manual_seed(SEED)


def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_text(pipe, prompt, max_tokens=MAX_NEW_TOKENS):
    """Generate text from prompt."""
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


def calculate_distinctness(texts, n=2):
    """Calculate distinct-n metric."""
    unique_ngrams = set()
    total_ngrams = 0
    
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
    
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0


def calculate_length_stats(texts):
    """Calculate average length statistics."""
    lengths = [len(text.split()) for text in texts]
    return {
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def run_evaluation():
    """Run baseline vs fine-tuned evaluation."""
    
    print("=" * 60)
    print("QuestCrafter Evaluation")
    print("=" * 60)
    
    # Test prompts
    test_prompts = [
        "A brave hero discovers a hidden temple",
        "In a magical forest, a young wizard learns to cast spells",
        "A merchant travels through a dangerous desert",
        "An ancient curse threatens the kingdom",
        "A group of adventurers finds a lost city",
    ]
    
    # Load baseline model
    print(f"\n1. Loading Baseline Model: {MODEL_NAME}")
    baseline_model, baseline_tokenizer = load_model_and_tokenizer(MODEL_NAME)
    baseline_pipe = pipeline(
        "text-generation",
        model=baseline_model,
        tokenizer=baseline_tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    
    baseline_generations = []
    baseline_texts = []
    
    print("\n2. Generating with Baseline Model...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"   [{i}/{len(test_prompts)}]", end=" ")
        generated = generate_text(baseline_pipe, prompt)
        baseline_generations.append({
            "prompt": prompt,
            "generated": generated,
        })
        baseline_texts.append(generated)
        print("✓")
    
    # Try to load fine-tuned model (if it exists)
    finetuned_generations = []
    finetuned_texts = []
    
    if FINETUNED_MODEL_DIR.exists():
        print(f"\n3. Loading Fine-tuned Model")
        try:
            finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(str(FINETUNED_MODEL_DIR))
            finetuned_pipe = pipeline(
                "text-generation",
                model=finetuned_model,
                tokenizer=finetuned_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            
            print("\n4. Generating with Fine-tuned Model...")
            for i, prompt in enumerate(test_prompts, 1):
                print(f"   [{i}/{len(test_prompts)}]", end=" ")
                generated = generate_text(finetuned_pipe, prompt)
                finetuned_generations.append({
                    "prompt": prompt,
                    "generated": generated,
                })
                finetuned_texts.append(generated)
                print("✓")
        except Exception as e:
            print(f"   ✗ Could not load fine-tuned model: {e}")
            finetuned_generations = None
    else:
        print(f"\n3. Fine-tuned model not found at {FINETUNED_MODEL_DIR}")
        print("   Run train.py to create a fine-tuned model")
    
    # Calculate metrics
    print("\n5. Calculating Metrics...")
    
    baseline_distinct = calculate_distinctness(baseline_texts)
    baseline_length = calculate_length_stats(baseline_texts)
    
    results = {
        "baseline": {
            "generations": baseline_generations,
            "metrics": {
                "distinct_2": baseline_distinct,
                "avg_length": baseline_length["avg_length"],
                "min_length": baseline_length["min_length"],
                "max_length": baseline_length["max_length"],
            }
        }
    }
    
    if finetuned_generations:
        finetuned_distinct = calculate_distinctness(finetuned_texts)
        finetuned_length = calculate_length_stats(finetuned_texts)
        
        results["finetuned"] = {
            "generations": finetuned_generations,
            "metrics": {
                "distinct_2": finetuned_distinct,
                "avg_length": finetuned_length["avg_length"],
                "min_length": finetuned_length["min_length"],
                "max_length": finetuned_length["max_length"],
            }
        }
    
    # Save results
    output_file = OUTPUTS_DIR / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"\nBaseline Metrics:")
    print(f"  Distinct-2:  {baseline_distinct:.3f}")
    print(f"  Avg Length:  {baseline_length['avg_length']:.1f} tokens")
    
    if finetuned_generations:
        print(f"\nFine-tuned Metrics:")
        print(f"  Distinct-2:  {finetuned_distinct:.3f}")
        print(f"  Avg Length:  {finetuned_length['avg_length']:.1f} tokens")
        print(f"\nImprovement:")
        print(f"  Distinct-2:  {(finetuned_distinct - baseline_distinct) * 100:+.1f}%")
    
    print(f"\n✓ Results saved to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
