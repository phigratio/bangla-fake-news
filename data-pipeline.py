import os
import time
import random
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================

class Config:
    # PASTE YOUR API KEY HERE
    API_KEY = "AIzaSyBvl6srlkHYj-7IygrE7ZASRSF_6ZEjO48"  
    
    MODEL_NAME = "gemma-3-27b-it" 
    
    # Augmentation Parameters
    N_GENERATIONS = 5
    K_SELECTION = 3
    
    # Similarity Filtering Threshold
    MIN_SIMILARITY = 0.7   
    
    # Prompting Strategy: 'zero-shot' or 'few-shot'
    PROMPTING_STRATEGY = "few-shot"
    
    # Subsampling Strategy: 'similarity' or 'random'
    SUBSAMPLING_STRATEGY = "random"
    
    # Dataset paths - Separate real and fake datasets
    REAL_DATASET_PATH = "LabeledAuthentic-7K.csv"
    FAKE_DATASET_PATH = "LabeledFake-1K.csv"
    
    # Train/Test split ratio
    TEST_SIZE = 0.3
    RANDOM_STATE = 50
    
    # Column names in your datasets
    CONTENT_COLUMN = "content"
    LABEL_COLUMN = "label"
    CATEGORY_COLUMN = "category" 

# Configure the Generative AI API
genai.configure(api_key=Config.API_KEY)

# ==========================================
# PHASE 1: PRE-PROCESSING
# ==========================================

class DataPreprocessor:
    @staticmethod
    def load_and_split_datasets():
        """
        Load separate real and fake datasets, split them individually,
        then combine for training and testing.
        Only FAKE data will be augmented. Real data stays as-is.
        """
        print("--- [Phase 1] Loading Datasets ---")
        
        # Load datasets
        df_real = pd.read_csv(Config.REAL_DATASET_PATH)
        df_fake = pd.read_csv(Config.FAKE_DATASET_PATH)
        
        print(f"Loaded {len(df_real)} real articles and {len(df_fake)} fake articles")
        
        # Keep only necessary columns
        df_fake = df_fake[['articleID', Config.CONTENT_COLUMN, Config.LABEL_COLUMN, Config.CATEGORY_COLUMN]].copy()
        df_fake['article_id'] = df_fake['articleID']
        
        # Split each dataset individually: 70% train, 30% test
        train_real, test_real = train_test_split(
            df_real, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE, 
            stratify=df_real[Config.LABEL_COLUMN]
        )
        
        print(f"Train REAL: {len(train_real)} articles")
        print(f"Train REAL category distribution:\n{train_real[Config.CATEGORY_COLUMN].value_counts()}\n")
        
        train_fake, test_fake = train_test_split(
            df_fake, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE, 
            stratify=df_fake[Config.LABEL_COLUMN]
        )
        
        print(f"Train FAKE: {len(train_fake)} articles")
        print(f"Train FAKE category distribution:\n{train_fake[Config.CATEGORY_COLUMN].value_counts()}\n")
        
        # Concatenate train splits and test splits
        train_df = pd.concat([train_real, train_fake]).sample(
            frac=1, 
            random_state=Config.RANDOM_STATE
        ).reset_index(drop=True)
        
        test_df = pd.concat([test_real, test_fake]).sample(
            frac=1, 
            random_state=Config.RANDOM_STATE
        ).reset_index(drop=True)
        
        # Print summary
        print(f"\n=== SPLIT SUMMARY ===")
        print(f"Total Train size: {len(train_df)} ({len(train_real)} real + {len(train_fake)} fake)")
        print(f"Total Test size: {len(test_df)} ({len(test_real)} real + {len(test_fake)} fake)")
        print(f"\nTrain label distribution:\n{train_df[Config.LABEL_COLUMN].value_counts()}")
        print(f"\nTest label distribution:\n{test_df[Config.LABEL_COLUMN].value_counts()}")
        
        return train_real, train_fake, test_df

# ==========================================
# PHASE 2: LLM AUGMENTATION ENGINE (Gemma)
# ==========================================

class AugmentationEngine:
    def __init__(self, model_name):
        # Initialize the Gemma model via API
        self.model = genai.GenerativeModel(model_name)
        self.output_path = f"raw_augmented_stream_K{Config.K_SELECTION}.csv"
        
        # Zero-Shot Prompt (for article-level paraphrasing)
        self.zero_shot_prompt = """Paraphrase the following news article delimited by triple backquotes in 5 different ways IN THE SAME LANGUAGE as the original. The generated articles must retain the exact same meaning, facts, label (real/fake), and LANGUAGE as the original article. DO NOT TRANSLATE - keep the same language. Maintain the article format and length. Enclose each paraphrased article within [BEGINARTICLE] and [ENDARTICLE] tags.
Only return the articles and no extra explanation.

```{text}```

NUMBERED GENERATED ARTICLES:"""
        
        # Few-Shot Prompt (for article-level paraphrasing with Bangla examples)
        self.few_shot_prompt = """Paraphrase the following news article delimited by triple backquotes in 5 different ways IN THE SAME LANGUAGE. The generated news articles must retain the exact same meaning and LANGUAGE as the original news article. DO NOT TRANSLATE. Enclose each paraphrased news article within [BEGINARTICLE] and [ENDARTICLE] tags.
Only return the paraphrases and no extra explanation.

Examples:
Provided Text: বাংলাদেশের মানুষের মাথাপিছু আয় এখন ২৫৫৪ ডলার যা ভারতের চেয়ে বেশি।
Paraphrases:
1. [BEGINARTICLE] বর্তমানে বাংলাদেশের জনগণের মাথাপিছু আয় ২৫৫৪ ডলারে পৌঁছেছে, যা ভারতের তুলনায় অধিক। [ENDARTICLE]
2. [BEGINARTICLE] বাংলাদেশের জনপ্রতি আয় এখন ২৫৫৪ ডলার, যা ভারতকে ছাড়িয়ে গেছে। [ENDARTICLE]
3. [BEGINARTICLE] ভারতের চেয়ে বেশি বাংলাদেশের মাথাপিছু আয় এখন ২৫৫৪ ডলার। [ENDARTICLE]
4. [BEGINARTICLE] বাংলাদেশীদের গড় আয় বর্তমানে ২৫৫৪ ডলার, যা ভারতের মাথাপিছু আয়ের চেয়ে বেশি। [ENDARTICLE]
5. [BEGINARTICLE] ২৫৫৪ ডলার মাথাপিছু আয় নিয়ে বাংলাদেশ এখন ভারতকে পেছনে ফেলেছে। [ENDARTICLE]

Provided Text: পদ্মা সেতুতে মানুষের কাটা মাথা লাগবে বলে গুজব ছড়িয়েছে একটি মহল।
Paraphrases:
1. [BEGINARTICLE] একটি মহল থেকে পদ্মা সেতু নির্মাণে মানুষের মাথা প্রয়োজন হবে এমন গুজব প্রচার করা হয়েছে। [ENDARTICLE]
2. [BEGINARTICLE] পদ্মা সেতু সম্পর্কে কাটা মাথা লাগবে বলে একটি গোষ্ঠী গুজব রটিয়েছে। [ENDARTICLE]
3. [BEGINARTICLE] কিছু লোক পদ্মা সেতুতে মানুষের মাথা ব্যবহার হবে এমন মিথ্যা খবর ছড়িয়ে দিয়েছে। [ENDARTICLE]

Now, paraphrase the following news article accordingly IN THE SAME LANGUAGE:
```{text}```

Paraphrases:"""

    def parse_generated_paraphrases(self, response_text):
        """
        Parse the generated text to extract paraphrases between [BEGINARTICLE] and [ENDARTICLE] tags.
        Returns a list of paraphrased articles.
        """
        import re
        paraphrases = re.findall(r'\[BEGINARTICLE\](.*?)\[ENDARTICLE\]', response_text, re.DOTALL)
        # Clean up whitespace
        paraphrases = [p.strip() for p in paraphrases if p.strip()]
        return paraphrases
    
    def flush_to_csv(self, buffer):
        """
        Atomic CSV writer: saves incrementally to prevent data loss
        """
        if not buffer:
            return
        
        df_flush = pd.DataFrame(buffer)
        
        # Check if file exists and has content
        write_header = not (os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0)
        
        # Append to CSV
        df_flush.to_csv(
            self.output_path,
            mode='a',
            index=False,
            header=write_header
        )
        
        print(f"[Checkpoint] Saved {len(df_flush)} rows → {self.output_path}")
    
    def generate_synthetic_data(self, train_fake_df):
        """
        Step 2.3: Generation loop for FAKE articles only WITH STREAMING SAVES.
        Generates N variants for each fake article with ID tracking (e.g., F1 -> F1a, F1b, F1c...)
        """
        print(f"\n{'='*60}")
        print(f"[Phase 2] STREAMING AUGMENTATION - SAVES EVERY 20 ROWS")
        print(f"{'='*60}")
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Prompting Strategy: {Config.PROMPTING_STRATEGY.upper()}")
        print(f"Output: {self.output_path}")
        print(f"Only augmenting {len(train_fake_df)} fake articles\n")
        
        batch_buffer = []
        rows_since_flush = 0
        
        # ---------- RESUME MODE ----------
        existing_ids = set()
        if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
            print(f"[RESUME MODE] Found existing file: {self.output_path}")
            existing_df = pd.read_csv(self.output_path)
            existing_ids = set(existing_df['original_id'].unique())
            print(f"Skipping {len(existing_ids)} already processed articles\n")
        
        # Select prompt based on strategy
        if Config.PROMPTING_STRATEGY == "zero-shot":
            prompt_template = self.zero_shot_prompt
        else:  # few-shot
            prompt_template = self.few_shot_prompt
        
        suffix_letters = 'abcdefghijklmnopqrstuvwxyz'
        
        # ------------------------------------------------------
        #                     MAIN LOOP
        # ------------------------------------------------------
        total_articles = len(train_fake_df)
        processed_count = 0
        
        # Iterate over FAKE training articles only
        for idx, row in train_fake_df.iterrows():
            # Use article content for augmentation
            original_article = row[Config.CONTENT_COLUMN]
            label = row[Config.LABEL_COLUMN]
            category = row[Config.CATEGORY_COLUMN]
            articleID = row['article_id']  # Original ID (e.g., F1, F2, ...)
            
            # Skip if already processed
            if articleID in existing_ids:
                print(f"[{idx+1}/{total_articles}] Skipping {articleID} (already processed)")
                continue
            
            print(f"[{idx+1}/{total_articles}] Processing {articleID}...")
            
            try:
                # Construct prompt
                prompt = prompt_template.format(text=original_article)
                
                # API Call to Gemma
                time.sleep(2) 
                response = self.model.generate_content(prompt)
                
                generated_text = response.text.strip()
                
                # Parse the paraphrases from the response
                paraphrases = self.parse_generated_paraphrases(generated_text)
                
                # If parsing fails, try to split by numbered lines as fallback
                if not paraphrases:
                    lines = generated_text.split('\n')
                    paraphrases = [line.strip() for line in lines if line.strip() and not line.strip().startswith(('1.', '2.', '3.', '4.', '5.', 'Paraphrases:', 'NUMBERED'))]
                
                if not paraphrases:
                    print(f"[Warning] No valid output for {articleID}")
                    continue
                
                # Check if paraphrases are in same language (simple check for Bangla)
                has_bangla_orig = any('\u0980' <= c <= '\u09FF' for c in original_article)
                has_bangla_para = any('\u0980' <= c <= '\u09FF' for c in paraphrases[0])
                
                if has_bangla_orig and not has_bangla_para:
                    print(f"[Warning] Model translated to English! Skipping {articleID}")
                    continue
                
                # Add each paraphrase to the augmented dataset with suffixed IDs
                # e.g., F1 -> F1a, F1b, F1c, F1d, F1e
                for i, para in enumerate(paraphrases[:Config.N_GENERATIONS]):
                    augmented_id = f"{articleID}{suffix_letters[i]}"
                    batch_buffer.append({
                        'article_id': augmented_id,
                        'original_id': articleID,
                        Config.CONTENT_COLUMN: para,
                        Config.LABEL_COLUMN: label,
                        Config.CATEGORY_COLUMN: category,
                        'is_augmented': True
                    })
                    rows_since_flush += 1
                
                print(f"Generated {len(paraphrases[:Config.N_GENERATIONS])} articles for {articleID}")
                processed_count += 1
                    
            except Exception as e:
                print(f"[API Error] Article {articleID}: {e}")
                # If API fails (quota limit), we skip this generation
                continue
            
            # ----------- PERIODIC FLUSH EVERY 20 ROWS -----------
            if rows_since_flush >= 20:
                self.flush_to_csv(batch_buffer)
                batch_buffer = []
                rows_since_flush = 0
        
        # ---------------- FINAL FLUSH ----------------
        if batch_buffer:
            self.flush_to_csv(batch_buffer)
        
        print(f"\n{'='*60}")
        print(f"STREAMING AUGMENTATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed: {processed_count} articles")
        print(f"Output saved to: {self.output_path}\n")
        
        # Load and return the complete dataset
        if os.path.exists(self.output_path):
            return pd.read_csv(self.output_path)
        else:
            return pd.DataFrame()

# ==========================================
# PHASE 3: SEMANTIC FILTERING (SBERT)
# ==========================================

class SemanticFilter:
    def __init__(self):
        # Load SBERT (Step 3.1)
        # 'all-MiniLM-L6-v2' is efficient and standard for this task
        print("--- Loading SBERT Model ---")
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')

    def filter_and_subsample(self, train_fake_df, augmented_df):
        """
        Step 3.2 & 3.3: Calculate Similarity and Subsample K
        Uses either similarity-based or random subsampling based on Config.SUBSAMPLING_STRATEGY
        Groups by original_id to process N variations of each article together
        """
        print(f"--- [Phase 3] Semantic Filtering & Subsampling ({Config.SUBSAMPLING_STRATEGY.upper()}) ---")
        print(f"--- Selecting best K={Config.K_SELECTION} from N={Config.N_GENERATIONS} generations ---")
        
        final_augmented_data = []
        
        # Group generated data by the original article ID to process N variations together
        grouped = augmented_df.groupby('original_id')
        
        for original_id, group in grouped:
            # Make a copy to avoid SettingWithCopyWarning
            group = group.copy()
            
            # Get the original article text from train_fake_df
            original_article = train_fake_df[train_fake_df['article_id'] == original_id][Config.CONTENT_COLUMN].iloc[0]
            
            if Config.SUBSAMPLING_STRATEGY == "similarity":
                # Similarity-based subsampling (as per paper)
                # 1. Encode Original Article
                orig_emb = self.sbert.encode(original_article, convert_to_tensor=True)
                
                # 2. Encode All Generated Candidates for this article
                cand_texts = group[Config.CONTENT_COLUMN].tolist()
                cand_embs = self.sbert.encode(cand_texts, convert_to_tensor=True)
                
                # 3. Calculate Cosine Similarity
                # Returns a list of scores matching the candidates
                cosine_scores = util.cos_sim(orig_emb, cand_embs)[0]
                
                # Add scores to the dataframe rows
                group['similarity_score'] = cosine_scores.cpu().numpy()
                
                # 4. Filter (Remove bad generations)
                valid_candidates = group[group['similarity_score'] > Config.MIN_SIMILARITY]
                
                # 5. Subsample (Select Top K by similarity)
                # Sort by similarity (descending) and take top K
                if not valid_candidates.empty:
                    best_k = valid_candidates.sort_values(
                        by='similarity_score', ascending=False
                    ).head(Config.K_SELECTION)
                    
                    final_augmented_data.append(best_k)
                    
            elif Config.SUBSAMPLING_STRATEGY == "random":
                # Pure random selection of K generated articles
                best_k = group.sample(n=min(Config.K_SELECTION, len(group)), random_state=42)
                final_augmented_data.append(best_k)
        
        if final_augmented_data:
            result_df = pd.concat(final_augmented_data)
            print(f"Selected {len(result_df)} augmented samples from {len(augmented_df)} generated samples")
            return result_df
        else:
            return pd.DataFrame()

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    print("\n" + "="*60)
    print("FAKE NEWS DATA AUGMENTATION PIPELINE")
    print("="*60)
    
    print(f"\n=== CONFIGURATION ===")
    print(f"Prompting Strategy: {Config.PROMPTING_STRATEGY.upper()}")
    print(f"Subsampling Strategy: {Config.SUBSAMPLING_STRATEGY.upper()}")
    print(f"Generations per article: {Config.N_GENERATIONS}")
    print(f"Best K to keep: {Config.K_SELECTION}")
    print(f"Min Similarity: {Config.MIN_SIMILARITY}")
    print(f"Test/Train Split: {Config.TEST_SIZE}/{1-Config.TEST_SIZE}")

    # 1. Load and Split Data (Real and Fake separately)
    train_real, train_fake, test_df = DataPreprocessor.load_and_split_datasets()

    # 2. Augmentation (Gemma) - ONLY augment FAKE articles
    print(f"\n{'='*60}")
    print("IMPORTANT: Only augmenting FAKE articles. Real articles kept as-is.")
    print(f"{'='*60}")
    augmentor = AugmentationEngine(Config.MODEL_NAME)
    raw_augmented_df = augmentor.generate_synthetic_data(train_fake)
    
    # 3. Subsampling / Filtering
    if Config.SUBSAMPLING_STRATEGY == "similarity":
        # Run SBERT-based filtering & top-K selection
        filter_engine = SemanticFilter()
        clean_augmented_df = filter_engine.filter_and_subsample(train_fake, raw_augmented_df)
    elif Config.SUBSAMPLING_STRATEGY == "random":
        # Randomly pick K articles per original without SBERT similarity
        final_rows = []
        grouped = raw_augmented_df.groupby('original_id')
        for original_id, group in grouped:
            if len(group) <= Config.K_SELECTION:
                selected = group
            else:
                selected = group.sample(n=Config.K_SELECTION, random_state=42)
            final_rows.append(selected)
        clean_augmented_df = pd.concat(final_rows).reset_index(drop=True)

    # 4. Combine Original Fake + Augmented Fake in natural order
    print(f"\n{'='*60}")
    print("COMBINING ORIGINAL + AUGMENTED FAKE ARTICLES (NATURAL ORDER)")
    print(f"{'='*60}")

    # Prepare original fake
    train_fake_prepared = train_fake[[
        'article_id', Config.CONTENT_COLUMN, Config.LABEL_COLUMN, Config.CATEGORY_COLUMN
    ]].copy()
    train_fake_prepared['is_augmented'] = False

    # Prepare augmented fake
    clean_augmented_df['is_augmented'] = True

    # Group augmented articles by original_id for ordering
    aug_grouped = clean_augmented_df.groupby('original_id')

    # Build final dataset: for each original, append original + its augmentations
    ordered_rows = []
    for idx, row in train_fake_prepared.iterrows():
        original_id = row['article_id']
        ordered_rows.append(row.to_dict())  # Convert to dict
        if original_id in aug_grouped.groups:
            aug_rows = aug_grouped.get_group(original_id)
            ordered_rows.extend(aug_rows.to_dict(orient='records'))  # append augmented

    # Convert to DataFrame
    fake_train_set = pd.DataFrame(ordered_rows)

    # Save CSV
    output_fake_train = f"fake_train_augmented_K{Config.K_SELECTION}.csv"
    fake_train_set.to_csv(output_fake_train, index=False)

    print(f"\nSaved fake training data (original + augmented in natural order) to: {output_fake_train}")
    print(f"Label distribution in fake training set:")
    print(fake_train_set[Config.LABEL_COLUMN].value_counts())
    print(f"Augmented vs Original in fake training set:")
    print(fake_train_set['is_augmented'].value_counts())

if __name__ == "__main__":
    run_pipeline()