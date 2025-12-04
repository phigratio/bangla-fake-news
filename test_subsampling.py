import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ==========================================
# TEST CONFIGURATION
# ==========================================

class TestConfig:
    # Test with 10 augmented rows (should be from 2-3 original articles)
    TEST_SAMPLE_SIZE = 10
    
    # Subsampling Strategy: 'similarity' or 'random'
    SUBSAMPLING_STRATEGY = "similarity"  # Change to "similarity" to test SBERT
    
    # Augmentation Parameters
    K_SELECTION = 3
    MIN_SIMILARITY = 0.7
    
    # Column names
    CONTENT_COLUMN = "content"
    LABEL_COLUMN = "label"
    CATEGORY_COLUMN = "category"
    
    # Paths
    AUGMENTED_DATA_PATH = "raw_augmented_stream_K3.csv"
    REAL_DATASET_PATH = "LabeledAuthentic-7K.csv"
    FAKE_DATASET_PATH = "LabeledFake-1K.csv"

# ==========================================
# SEMANTIC FILTER (Same as main pipeline)
# ==========================================

class SemanticFilter:
    def __init__(self):
        print("--- Loading SBERT Model ---")
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')

    def filter_and_subsample(self, train_fake_df, augmented_df):
        """
        Calculate Similarity and Subsample K
        Uses either similarity-based or random subsampling based on TestConfig.SUBSAMPLING_STRATEGY
        """
        print(f"\n--- [Phase 3] Semantic Filtering & Subsampling ({TestConfig.SUBSAMPLING_STRATEGY.upper()}) ---")
        print(f"--- Selecting best K={TestConfig.K_SELECTION} from augmented samples ---")
        
        final_augmented_data = []
        
        # Group generated data by the original article ID
        grouped = augmented_df.groupby('original_id')
        
        print(f"\nFound {len(grouped)} unique original articles in test sample")
        
        for original_id, group in grouped:
            # Make a copy to avoid SettingWithCopyWarning
            group = group.copy()
            
            print(f"\nProcessing original_id: {original_id} ({len(group)} augmented versions)")
            
            # Get the original article text from train_fake_df
            matching_articles = train_fake_df[train_fake_df['article_id'] == original_id]
            
            if matching_articles.empty:
                print(f"  WARNING: Original article {original_id} not found in train_fake_df")
                continue
                
            original_article = matching_articles[TestConfig.CONTENT_COLUMN].iloc[0]
            
            if TestConfig.SUBSAMPLING_STRATEGY == "similarity":
                # Similarity-based subsampling
                print(f"  Using SIMILARITY-based selection...")
                
                # 1. Encode Original Article
                orig_emb = self.sbert.encode(original_article, convert_to_tensor=True)
                
                # 2. Encode All Generated Candidates for this article
                cand_texts = group[TestConfig.CONTENT_COLUMN].tolist()
                cand_embs = self.sbert.encode(cand_texts, convert_to_tensor=True)
                
                # 3. Calculate Cosine Similarity
                cosine_scores = util.cos_sim(orig_emb, cand_embs)[0]
                
                # Add scores to the dataframe rows
                group['similarity_score'] = cosine_scores.cpu().numpy()
                
                print(f"  Similarity scores: {cosine_scores.cpu().numpy()}")
                
                # 4. Filter (Remove bad generations)
                valid_candidates = group[group['similarity_score'] > TestConfig.MIN_SIMILARITY]
                print(f"  Valid candidates (similarity > {TestConfig.MIN_SIMILARITY}): {len(valid_candidates)}")
                
                # 5. Subsample (Select Top K by similarity)
                if not valid_candidates.empty:
                    best_k = valid_candidates.sort_values(
                        by='similarity_score', ascending=False
                    ).head(TestConfig.K_SELECTION)
                    
                    print(f"  Selected top {len(best_k)} samples")
                    final_augmented_data.append(best_k)
                else:
                    print(f"  WARNING: No valid candidates for {original_id}")
                    
            elif TestConfig.SUBSAMPLING_STRATEGY == "random":
                # Pure random selection of K generated articles
                print(f"  Using RANDOM selection...")
                best_k = group.sample(n=min(TestConfig.K_SELECTION, len(group)), random_state=42)
                print(f"  Randomly selected {len(best_k)} samples")
                final_augmented_data.append(best_k)
        
        if final_augmented_data:
            result_df = pd.concat(final_augmented_data)
            print(f"\n=== SUBSAMPLING COMPLETE ===")
            print(f"Total selected samples: {len(result_df)}")
            print(f"Original samples processed: {len(grouped)}")
            return result_df
        else:
            print(f"\n=== WARNING: No data selected ===")
            return pd.DataFrame()

# ==========================================
# TEST EXECUTION
# ==========================================

def test_subsampling():
    print("\n" + "="*60)
    print("TESTING SUBSAMPLING PIPELINE WITH 10 ROWS")
    print("="*60)
    
    print(f"\n=== TEST CONFIGURATION ===")
    print(f"Subsampling Strategy: {TestConfig.SUBSAMPLING_STRATEGY.upper()}")
    print(f"Test Sample Size: {TestConfig.TEST_SAMPLE_SIZE} rows")
    print(f"Best K to keep per original: {TestConfig.K_SELECTION}")
    print(f"Min Similarity Threshold: {TestConfig.MIN_SIMILARITY}")
    
    # 1. Load augmented data
    print(f"\n--- Loading Augmented Data ---")
    augmented_df = pd.read_csv(TestConfig.AUGMENTED_DATA_PATH)
    print(f"Total augmented rows available: {len(augmented_df)}")
    
    # Sample 10 rows for testing
    test_sample = augmented_df.head(TestConfig.TEST_SAMPLE_SIZE).copy()
    print(f"Test sample size: {len(test_sample)}")
    print(f"\nTest sample info:")
    print(test_sample[['article_id', 'original_id', TestConfig.LABEL_COLUMN, TestConfig.CATEGORY_COLUMN]].to_string())
    
    # Check unique original_ids
    unique_originals = test_sample['original_id'].unique()
    print(f"\nUnique original article IDs in test sample: {len(unique_originals)}")
    print(f"Original IDs: {unique_originals}")
    
    # 2. Load the original fake dataset to get original articles
    print(f"\n--- Loading Original Fake Dataset ---")
    df_fake = pd.read_csv(TestConfig.FAKE_DATASET_PATH)
    df_fake = df_fake[['articleID', TestConfig.CONTENT_COLUMN, TestConfig.LABEL_COLUMN, TestConfig.CATEGORY_COLUMN]].copy()
    df_fake['article_id'] = df_fake['articleID']
    print(f"Loaded {len(df_fake)} fake articles")
    
    # Filter to only the original articles we need for the test
    train_fake_subset = df_fake[df_fake['article_id'].isin(unique_originals)]
    print(f"Filtered to {len(train_fake_subset)} original articles matching test sample")
    
    if len(train_fake_subset) == 0:
        print("\nERROR: No matching original articles found!")
        print("This might be because the original_id format doesn't match articleID format")
        return
    
    # 3. Run subsampling
    if TestConfig.SUBSAMPLING_STRATEGY == "similarity":
        filter_engine = SemanticFilter()
        result_df = filter_engine.filter_and_subsample(train_fake_subset, test_sample)
    else:
        # Random subsampling without loading SBERT
        print(f"\n--- [Phase 3] Random Subsampling ---")
        final_rows = []
        grouped = test_sample.groupby('original_id')
        for original_id, group in grouped:
            print(f"\nProcessing original_id: {original_id} ({len(group)} augmented versions)")
            if len(group) <= TestConfig.K_SELECTION:
                selected = group
                print(f"  Keeping all {len(selected)} samples (less than K={TestConfig.K_SELECTION})")
            else:
                selected = group.sample(n=TestConfig.K_SELECTION, random_state=42)
                print(f"  Randomly selected {len(selected)} samples")
            final_rows.append(selected)
        result_df = pd.concat(final_rows).reset_index(drop=True)
        print(f"\n=== RANDOM SUBSAMPLING COMPLETE ===")
        print(f"Total selected samples: {len(result_df)}")
    
    # 4. Display results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\nInput test sample: {len(test_sample)} rows")
    print(f"Output after subsampling: {len(result_df)} rows")
    print(f"\nFinal selected samples:")
    print(result_df[['article_id', 'original_id', TestConfig.LABEL_COLUMN, TestConfig.CATEGORY_COLUMN]].to_string())
    
    # Save test output
    output_file = f"test_subsampled_K{TestConfig.K_SELECTION}.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\n✓ Test output saved to: {output_file}")
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE - Subsampling pipeline is working!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_subsampling()
