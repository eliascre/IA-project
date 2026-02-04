# QuestCrafter Quick Start Guide

## 🚀 Immediate Next Steps

### Step 1: Install Dependencies (5 minutes)

```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import torch; import transformers; print('✓ All packages installed')"
```

---

### Step 2: Preprocess Your Data (2 minutes)

Convert raw CSV → instruction-response format and create data splits:

```bash
python preprocess.py
```

**Expected output:**
```
✓ Loaded 1000 stories from train.csv
✓ Created 1000 instruction-response pairs
✓ Saved 800 records to data/train.jsonl
✓ Saved 100 records to data/val.jsonl
✓ Saved 100 records to data/test.jsonl
```

**Check results:**
```bash
ls -la data/
cat data/train.jsonl | head -3
```

---

### Step 3: Run Baseline (Optional but Recommended - 5 minutes)

See how the pretrained model performs **before** fine-tuning:

```bash
python baseline.py
```

**This helps you:**
- Understand typical failure modes
- Set expectations for fine-tuning improvements
- Create a comparison point

**Output:** `outputs/baseline_generations.jsonl`

---

### Step 4: Fine-tune the Model (30-60 minutes)

Train your model on the quest dataset:

```bash
python train.py
```

**What happens:**
1. Loads `distilgpt2` (or your configured model)
2. Trains on `data/train.jsonl` for 3 epochs
3. Validates on `data/val.jsonl` each epoch
4. Saves best model to `models/quest_crafter_finetuned/`
5. Saves training logs to `outputs/training_logs/`

**Monitor training:**
```bash
# In another terminal, watch training logs
tail -f outputs/training_logs/trainer_state.json
```

**Expect:**
- Epoch 1: Loss ~4.5 → 3.2
- Epoch 2: Loss ~3.0 → 2.5
- Epoch 3: Loss ~2.3 → 2.0

---

### Step 5: Evaluate Results (5 minutes)

Compare baseline vs fine-tuned:

```bash
python evaluate.py
```

**Output:** `outputs/evaluation_results.json`

**Sample results:**
```
Baseline Metrics:
  Distinct-2:  0.450
  Avg Length:  185 tokens

Fine-tuned Metrics:
  Distinct-2:  0.580  (+28.9%)
  Avg Length:  210 tokens

✓ Fine-tuning improves diversity!
```

---

### Step 6: Launch Demo App (2 minutes)

Interactive quest generation:

```bash
streamlit run demo_app.py
```

**Then open:** http://localhost:8501

**Try it:**
1. Select "Fine-tuned" model (or "Baseline" for comparison)
2. Adjust Temperature (0.1-2.0) and Top-P (0-1.0)
3. Enter a prompt or click "Use Example"
4. Click "✨ Generate Quest"

---

## 📊 Project Timeline

| Week | Focus | Key Output |
|------|-------|-----------|
| W1 | Data preprocessing ✓ | train/val/test splits |
| W2 | Baseline + analysis | baseline_generations.jsonl |
| W3 | Fine-tuning (you are here) | quest_crafter_finetuned/ |
| W4 | Evaluation & control | evaluation_results.json |
| W5 | Demo + report | README + demo_app.py |

---

## 🎯 Current Progress Checklist

- [x] Project structure created
- [x] Configuration file (`config.py`)
- [x] Data pipeline script (`preprocess.py`)
- [x] Baseline inference (`baseline.py`)
- [x] Training script (`train.py`)
- [x] Evaluation script (`evaluate.py`)
- [x] Demo app (`demo_app.py`)
- [ ] **NEXT:** Run `python preprocess.py`
- [ ] Run `python train.py`
- [ ] Run `python evaluate.py`
- [ ] Test demo app

---

## 📁 File Structure Created

```
c:\Users\elias_nrng862\Documents\AI Project\
├── config.py                    ← Global settings
├── preprocess.py                ← Data pipeline
├── baseline.py                  ← Pretrained model
├── train.py                     ← Fine-tuning
├── evaluate.py                  ← Metrics
├── demo_app.py                  ← Streamlit app
├── evaluation_rubric.py          ← Human scoring guide
├── progress_tracker.py           ← Timeline tracker
├── requirements.txt             ← Dependencies
├── README.md                    ← Full documentation
│
├── data/                        ← Data splits (created after preprocess)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
│
├── models/                      ← Fine-tuned models (created after train)
│   └── quest_crafter_finetuned/
│
├── outputs/                     ← Results
│   ├── baseline_generations.jsonl
│   ├── evaluation_results.json
│   └── training_logs/
│
└── notebooks/                   ← Optional Jupyter analysis
```

---

## 🛠️ Customization Tips

**Want to adjust settings?** Edit `config.py`:

```python
# Use faster model
MODEL_NAME = "distilgpt2"

# Train longer
EPOCHS = 5

# Smaller batch size (if memory error)
BATCH_SIZE = 4

# Generate longer quests
MAX_NEW_TOKENS = 300
```

---

## ⚠️ Common Issues & Solutions

### Issue: "CUDA out of memory"
```python
# In config.py, reduce:
BATCH_SIZE = 4  # was 8
MAX_LENGTH = 128  # was 256
```

### Issue: "Module not found: transformers"
```bash
pip install --upgrade transformers
```

### Issue: No data files found
```bash
# Make sure these exist:
ls train.csv
# Then run preprocess
python preprocess.py
```

### Issue: Demo app won't load fine-tuned model
```bash
# First train the model:
python train.py
# Then try demo again
streamlit run demo_app.py
```

---

## 📚 For Your Final Report

Make sure to include:

1. **Dataset Description**
   - Number of stories: ~1000
   - Average length: ~200 tokens
   - Train/val/test split: 80/10/10

2. **Baseline Analysis**
   - Model: `distilgpt2`
   - Diversity (Distinct-2): 0.45
   - Common issues: repetition, off-topic

3. **Fine-tuning Setup**
   - Model: `distilgpt2`
   - Epochs: 3
   - Learning rate: 5e-5
   - Batch size: 8

4. **Results**
   - Training curves (loss over epochs)
   - Comparison table (baseline vs fine-tuned)
   - Sample generations
   - Human evaluation scores

5. **Lessons Learned**
   - What worked well
   - What didn't work
   - Future improvements

---

## 🎓 Learning Objectives Achieved

By completing this project, you will have:

1. ✓ Prepared a supervised dataset (JSONL/CSV)
2. ✓ Run inference with pretrained model
3. ✓ Fine-tuned a model with PyTorch
4. ✓ Evaluated with automatic metrics + human rubric
5. ✓ Built a reproducible demo app

---

## 💡 Pro Tips

1. **Start with baseline**: Always run `baseline.py` first to understand issues
2. **Use small batches**: Start with `BATCH_SIZE=4` to avoid OOM
3. **Track everything**: Save model checkpoints and training logs
4. **Evaluate often**: Run `evaluate.py` after each training run
5. **Keep examples**: Save good/bad generations for your report

---

## Next: Run the Pipeline!

Ready? Start with step 1:

```bash
pip install -r requirements.txt
python preprocess.py
python train.py
python evaluate.py
streamlit run demo_app.py
```

Good luck! 🚀
