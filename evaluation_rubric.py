"""
Human Evaluation Rubric for QuestCrafter Generations
Use this to score generated quests on a 1-5 scale
"""

EVALUATION_RUBRIC = {
    "coherence": {
        "description": "Logical story flow, no abrupt jumps or nonsensical transitions",
        "scoring": {
            1: "Completely incoherent, random sentences, no narrative flow",
            2: "Mostly incoherent with some meaningful sentences",
            3: "Somewhat coherent with occasional logical gaps",
            4: "Mostly coherent with logical progression",
            5: "Excellent coherence, clear narrative arc and smooth transitions"
        }
    },
    
    "prompt_faithfulness": {
        "description": "Respects the input prompt's constraints, themes, and requirements",
        "scoring": {
            1: "Completely ignores prompt, unrelated topic",
            2: "Partially follows prompt, many deviations",
            3: "Follows prompt partially, some relevant elements",
            4: "Follows prompt well with minor deviations",
            5: "Perfectly captures prompt requirements and theme"
        }
    },
    
    "creativity": {
        "description": "Interesting elements, unique ideas, not generic or overly repetitive",
        "scoring": {
            1: "Extremely generic or repetitive, no creative elements",
            2: "Mostly generic with minimal creative additions",
            3: "Some creative elements mixed with generic content",
            4: "Notably creative with interesting ideas",
            5: "Highly creative and original, engaging storyline"
        }
    },
    
    "length": {
        "description": "Appropriate length - not too short (<50 tokens) or too long (>300 tokens)",
        "scoring": {
            1: "Extremely short (<20 tokens) or unreasonably long (>500 tokens)",
            2: "Too short (<50 tokens) or too long (>350 tokens)",
            3: "Slightly off ideal range (40-70 or 250-350 tokens)",
            4: "Good length (70-250 tokens), minor issues",
            5: "Perfect length (80-240 tokens), well-paced"
        }
    },
    
    "grammar_clarity": {
        "description": "Well-formed sentences, proper grammar, easy to understand",
        "scoring": {
            1: "Many grammatical errors, hard to understand",
            2: "Several errors, somewhat unclear",
            3: "Some errors but mostly understandable",
            4: "Minor errors, generally clear",
            5: "Excellent grammar and clarity throughout"
        }
    }
}


def print_rubric():
    """Print the evaluation rubric."""
    print("=" * 70)
    print("QUESTCRAFTER HUMAN EVALUATION RUBRIC")
    print("=" * 70)
    
    for criterion, details in EVALUATION_RUBRIC.items():
        print(f"\n{criterion.upper()}")
        print(f"Description: {details['description']}")
        print("Scoring guide:")
        for score, description in details['scoring'].items():
            print(f"  {score}/5: {description}")
    
    print("\n" + "=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print("""
1. Read each generated quest carefully
2. Score each criterion on 1-5 scale
3. Provide brief comments for clarification
4. Calculate AVERAGE of 5 criteria = OVERALL SCORE
5. Compare baseline vs fine-tuned

Example evaluation:
  Prompt: "A brave knight must rescue a princess from a dark tower"
  
  Coherence:          4/5 (good flow, minor jump at end)
  Prompt-Faithfulness: 5/5 (perfectly captures quest theme)
  Creativity:         3/5 (somewhat generic knight story)
  Length:             4/5 (well-paced, 180 tokens)
  Grammar:            5/5 (excellent, no errors)
  
  OVERALL:            4.2/5
  Comments: Good quest but could be more original
    """)


def sample_scoring_template():
    """Return a template for evaluating multiple generations."""
    return """
QUESTCRAFTER EVALUATION FORM
==============================

Test Prompt: ___________________________________

Evaluator: ________________  Date: ____________

BASELINE MODEL ({model_name})
-----------------------------------
Coherence:           ___/5
Prompt-Faithfulness: ___/5
Creativity:          ___/5
Length:              ___/5
Grammar/Clarity:     ___/5
AVERAGE:             ___/5

Comments: _________________________________

FINE-TUNED MODEL
-----------------------------------
Coherence:           ___/5
Prompt-Faithfulness: ___/5
Creativity:          ___/5
Length:              ___/5
Grammar/Clarity:     ___/5
AVERAGE:             ___/5

Comments: _________________________________

OVERALL WINNER: [ ] Baseline [ ] Fine-tuned [ ] Tie
Reason: _____________________________________
    """


if __name__ == "__main__":
    print_rubric()
