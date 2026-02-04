"""
QuestCrafter Streamlit Demo App
Interactive quest generation interface
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from config import (
    MODEL_NAME,
    FINETUNED_MODEL_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)


@st.cache_resource
def load_baseline_model():
    """Load baseline pretrained model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    
    return pipe


@st.cache_resource
def load_finetuned_model():
    """Load fine-tuned model."""
    if not FINETUNED_MODEL_DIR.exists():
        return None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_MODEL_DIR))
        model = AutoModelForCausalLM.from_pretrained(str(FINETUNED_MODEL_DIR))
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        
        return pipe
    except Exception as e:
        st.warning(f"Could not load fine-tuned model: {e}")
        return None


def generate_quest(pipe, prompt, max_tokens, temperature, top_p):
    """Generate a quest using the pipeline."""
    output = pipe(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=TOP_K,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
    )
    return output[0]['generated_text']


def main():
    st.set_page_config(
        page_title="QuestCrafter",
        page_icon="⚔️",
        layout="wide"
    )
    
    st.title("⚔️ QuestCrafter")
    st.write("Generate fantasy quests with AI-powered storytelling")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    model_choice = st.sidebar.radio(
        "Choose Model",
        ["Baseline (Pretrained)", "Fine-tuned"],
        help="Select between pretrained and fine-tuned model"
    )
    
    max_tokens = st.sidebar.slider(
        "Max Response Length",
        min_value=50,
        max_value=500,
        value=200,
        step=10,
    )
    
    temperature = st.sidebar.slider(
        "Temperature (Creativity)",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Higher = more creative, Lower = more predictable"
    )
    
    top_p = st.sidebar.slider(
        "Top-P (Diversity)",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
    )
    
    # Load models
    with st.spinner("Loading model..."):
        if model_choice == "Baseline (Pretrained)":
            pipe = load_baseline_model()
            model_name = f"Baseline: {MODEL_NAME}"
        else:
            pipe = load_finetuned_model()
            if pipe is None:
                st.error("Fine-tuned model not found. Please train a model first.")
                return
            model_name = "Fine-tuned Model"
    
    st.sidebar.success(f"✓ Model loaded: {model_name}")
    
    # Main interface
    st.subheader("📖 Quest Prompt")
    
    # Example prompts
    example_prompts = [
        "A brave knight must rescue a princess from a dark tower",
        "In a magical forest, a young mage discovers an ancient spell",
        "A thief in a bustling city discovers a hidden treasure map",
        "An adventurer wakes up in a mysterious dungeon",
        "A group of friends must stop an evil sorcerer",
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_prompt = st.text_input(
            "Enter your quest prompt:",
            placeholder="E.g., A brave hero in a dark forest..."
        )
    
    with col2:
        use_example = st.button("📚 Use Example")
        if use_example:
            user_prompt = st.session_state.get('example_prompt', example_prompts[0])
    
    # Quick select examples
    st.write("**Quick Examples:**")
    cols = st.columns(len(example_prompts))
    for i, (col, example) in enumerate(zip(cols, example_prompts)):
        with col:
            if st.button(f"#{i+1}", key=f"example_{i}"):
                st.session_state.example_prompt = example
                st.rerun()
    
    # Generate button
    if st.button("✨ Generate Quest", type="primary", use_container_width=True):
        if not user_prompt:
            st.error("Please enter a quest prompt!")
        else:
            with st.spinner("Generating your quest..."):
                try:
                    generated_text = generate_quest(
                        pipe,
                        user_prompt,
                        max_tokens,
                        temperature,
                        top_p
                    )
                    
                    st.subheader("⚔️ Generated Quest")
                    st.text_area(
                        "Result:",
                        value=generated_text,
                        height=250,
                        disabled=True
                    )
                    
                    # Copy button
                    st.button("📋 Copy to Clipboard")
                    
                except Exception as e:
                    st.error(f"Error generating quest: {e}")
    
    # Info section
    st.divider()
    st.subheader("ℹ️ About QuestCrafter")
    st.write("""
    **QuestCrafter** uses advanced language models to generate engaging fantasy quests.
    
    - **Baseline Model**: Pretrained model without fine-tuning
    - **Fine-tuned Model**: Optimized on fantasy quest data
    - **Temperature**: Controls the randomness of generations (higher = more creative)
    - **Top-P**: Controls diversity of word selection
    
    Generated quests may vary in quality and length. Use the settings to customize output!
    """)


if __name__ == "__main__":
    main()
