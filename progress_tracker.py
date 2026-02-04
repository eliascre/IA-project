"""
QuestCrafter Project Progress Tracker
Track your progress through the 5-week project timeline
"""

PROJECT_TIMELINE = {
    "Week 1": {
        "focus": "Dataset selection + exploration; cleaning and formatting",
        "tasks": [
            ("Download/select dataset", False),
            ("Explore data structure", False),
            ("Create preprocessing pipeline", False),
            ("Define quest template fields", False),
            ("Build train/val/test splits", False),
        ],
        "outputs": [
            "train.jsonl, val.jsonl, test.jsonl",
            "data statistics report",
        ]
    },
    
    "Week 2": {
        "focus": "Baseline generation and evaluation setup",
        "tasks": [
            ("Run pretrained model on test prompts", False),
            ("Create 50+ test prompts covering diverse settings", False),
            ("Document common failure modes", False),
            ("Define human evaluation rubric", False),
            ("First failure analysis", False),
        ],
        "outputs": [
            "baseline_generations.jsonl",
            "failure_analysis_report.txt",
            "test_prompts.txt",
        ]
    },
    
    "Week 3": {
        "focus": "Fine-tuning with PyTorch",
        "tasks": [
            ("Implement training script", False),
            ("Train model on 1-3 epochs", False),
            ("Track training curves (loss, metrics)", False),
            ("Save checkpoints", False),
            ("Sample and analyze trained outputs", False),
        ],
        "outputs": [
            "models/quest_crafter_finetuned/",
            "training_logs/",
            "sample_outputs_after_training.jsonl",
        ]
    },
    
    "Week 4": {
        "focus": "Control, safety, and robust evaluation",
        "tasks": [
            ("Add control tokens (LEVEL, SETTING, TONE)", False),
            ("Implement length/quality filters", False),
            ("Run structured evaluation: baseline vs tuned", False),
            ("Test robustness with adversarial prompts", False),
            ("Compile evaluation metrics table", False),
        ],
        "outputs": [
            "evaluation_results.json",
            "baseline_vs_finetuned_comparison.csv",
            "robustness_report.txt",
        ]
    },
    
    "Week 5": {
        "focus": "Demo, documentation, and final delivery",
        "tasks": [
            ("Build Streamlit/Gradio demo app", False),
            ("Write final project report (4-8 pages)", False),
            ("Create clean repo with README", False),
            ("Prepare presentation slides", False),
            ("Final testing and bug fixes", False),
        ],
        "outputs": [
            "demo_app.py",
            "final_report.pdf",
            "README.md",
            "presentation.pptx",
        ]
    }
}


def print_timeline():
    """Print the project timeline."""
    print("=" * 70)
    print("QUESTCRAFTER 5-WEEK PROJECT TIMELINE")
    print("=" * 70)
    
    for week, details in PROJECT_TIMELINE.items():
        print(f"\n{week}")
        print(f"Focus: {details['focus']}")
        print(f"Tasks:")
        for task, done in details['tasks']:
            status = "✓" if done else "○"
            print(f"  {status} {task}")
        print(f"Outputs:")
        for output in details['outputs']:
            print(f"    - {output}")


def get_current_week_tasks(week_number):
    """Get tasks for a specific week."""
    weeks = list(PROJECT_TIMELINE.keys())
    if week_number > len(weeks) or week_number < 1:
        return None
    
    week_key = weeks[week_number - 1]
    return PROJECT_TIMELINE[week_key]


if __name__ == "__main__":
    print_timeline()
    
    print("\n" + "=" * 70)
    print("MANDATORY DELIVERABLES")
    print("=" * 70)
    print("""
1. Dataset Pipeline
   - Download/clean data
   - Format to JSONL (instruction → response)
   - Create train/val/test splits (80/10/10)

2. Baseline Generation
   - Run pretrained model on test set
   - Document issues: repetition, off-topic, length, coherence

3. Fine-tuning with PyTorch
   - Train on distilgpt2/gpt2/t5-small
   - Document: epochs, LR, batch size, max length
   - Save checkpoints

4. Evaluation
   - Test set: 50+ diverse prompts
   - Automatic: validation loss, Distinct-n metric
   - Human: coherence, creativity, prompt-faithfulness (1-5)
   - Comparison: baseline vs fine-tuned

5. Final Deliverable
   - Git repo with clean structure
   - README with instructions
   - Report (4-8 pages): dataset, methods, results
   - Demo app (Streamlit/Gradio)
    """)
