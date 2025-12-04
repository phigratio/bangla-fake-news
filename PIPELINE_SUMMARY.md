# Data Augmentation Pipeline - Summary

## Sample Datasets Created:
1. **sample_fake_5.csv** - 5 fake news articles in Bangla
2. **sample_real_2.csv** - 2 real news articles in Bangla

## Key Updates:

### 1. Prompts (English with Strict Bangla Enforcement)
- **Zero-Shot Prompt**: English instructions with clear emphasis on Bangla-only output
- **Few-Shot Prompt**: English instructions with Bangla examples showing proper paraphrasing
- Both prompts include multiple warnings: "DO NOT use English", "STRICTLY OUTPUT IN BANGLA", "100% in Bangla"

### 2. Configuration
- Using sample files for testing: `sample_fake_5.csv` and `sample_real_2.csv`
- N_GENERATIONS = 5 (generate 5 paraphrases per article)
- K_SELECTION = 3 (keep best 3 after filtering)
- Strategies: zero-shot and few-shot (can switch between them)
- Subsampling: random (can also use similarity-based)

### 3. ID System
- Real articles: R0, R1, R2...
- Fake articles (original): F0, F1, F2, F3, F4
- Augmented fake articles: F0a, F0b, F0c (from F0), F1a, F1b, F1c (from F1), etc.

### 4. Pipeline Flow
```
1. Load 5 fake + 2 real articles
2. Split: ~3 fake train, ~2 fake test, ~1 real train, ~1 real test
3. Augment ONLY the 3 fake training articles (Bangla to Bangla)
4. Generate 5 paraphrases per article = 15 total generations
5. Filter by similarity (>0.7) and select best 3 per article = ~9 augmented articles
6. Combine: 1 real + 3 original fake + 9 augmented fake = 13 training samples
7. Test set: ~3 articles (mix of real and fake, no augmentation)
```

### 5. Output Files
- `final_train_augmented_K3.csv` - Training data with augmented articles
- `final_test_original.csv` - Test data (no augmentation)

### 6. Testing Both Strategies
Run `run_both_strategies.py` to test both zero-shot and few-shot prompting automatically.

## Running the Pipeline

### Single run (zero-shot):
```bash
python data-pipeline.py
```

### Test both strategies:
```bash
python run_both_strategies.py
```

## Expected Behavior
- All augmented articles will be in Bangla (no English translation)
- Each fake article will generate 3 high-quality paraphrases
- IDs will be properly tracked (F1 -> F1a, F1b, F1c)
- Real articles remain unchanged in the training set
