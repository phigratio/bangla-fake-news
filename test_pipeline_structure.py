"""
Test script to verify the pipeline structure without running the full augmentation.
This checks that all the data loading and ID assignment works correctly.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
REAL_DATASET_PATH = "LabeledAuthentic-7K.csv"
FAKE_DATASET_PATH = "LabeledFake-1K.csv"
CONTENT_COLUMN = "content"
LABEL_COLUMN = "label"
CATEGORY_COLUMN = "category"
TEST_SIZE = 0.3
RANDOM_STATE = 50

def test_data_loading():
    print("Testing data loading and splitting...")
    
    try:
        # Load datasets
        df_real = pd.read_csv(REAL_DATASET_PATH)
        df_fake = pd.read_csv(FAKE_DATASET_PATH)
        
        print(f"✓ Loaded {len(df_real)} real articles")
        print(f"✓ Loaded {len(df_fake)} fake articles")
        
        # Check columns
        print(f"\nReal dataset columns: {df_real.columns.tolist()}")
        print(f"Fake dataset columns: {df_fake.columns.tolist()}")
        
        # Keep only necessary columns
        df_real = df_real[[CONTENT_COLUMN, LABEL_COLUMN, CATEGORY_COLUMN]].copy()
        df_fake = df_fake[[CONTENT_COLUMN, LABEL_COLUMN, CATEGORY_COLUMN]].copy()
        
        # Add IDs
        df_real['article_id'] = ['R' + str(i) for i in range(len(df_real))]
        df_fake['article_id'] = ['F' + str(i) for i in range(len(df_fake))]
        
        print(f"\n✓ Added article IDs")
        print(f"Sample real IDs: {df_real['article_id'].head(3).tolist()}")
        print(f"Sample fake IDs: {df_fake['article_id'].head(3).tolist()}")
        
        # Split datasets
        train_real, test_real = train_test_split(
            df_real, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
            stratify=df_real[LABEL_COLUMN]
        )
        
        train_fake, test_fake = train_test_split(
            df_fake, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
            stratify=df_fake[LABEL_COLUMN]
        )
        
        print(f"\n✓ Split completed:")
        print(f"  Train REAL: {len(train_real)} articles")
        print(f"  Test REAL: {len(test_real)} articles")
        print(f"  Train FAKE: {len(train_fake)} articles")
        print(f"  Test FAKE: {len(test_fake)} articles")
        
        # Test ID augmentation simulation
        print(f"\n✓ Simulating ID augmentation:")
        sample_id = train_fake['article_id'].iloc[0]
        suffix_letters = 'abcdefghijklmnopqrstuvwxyz'
        augmented_ids = [f"{sample_id}{suffix_letters[i]}" for i in range(5)]
        print(f"  Original ID: {sample_id}")
        print(f"  Augmented IDs: {augmented_ids}")
        
        # Combine datasets
        train_df = pd.concat([train_real, train_fake]).sample(
            frac=1, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        
        test_df = pd.concat([test_real, test_fake]).sample(
            frac=1, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        
        print(f"\n✓ Combined datasets:")
        print(f"  Total Train: {len(train_df)} ({len(train_real)} real + {len(train_fake)} fake)")
        print(f"  Total Test: {len(test_df)} ({len(test_real)} real + {len(test_fake)} fake)")
        
        print(f"\n✓ Label distribution in train:")
        print(train_df[LABEL_COLUMN].value_counts())
        
        print(f"\n✓ Category distribution in train REAL:")
        print(train_real[CATEGORY_COLUMN].value_counts())
        
        print(f"\n✓ Category distribution in train FAKE:")
        print(train_fake[CATEGORY_COLUMN].value_counts())
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Dataset file not found - {e}")
        print("Please ensure the following files exist:")
        print(f"  - {REAL_DATASET_PATH}")
        print(f"  - {FAKE_DATASET_PATH}")
    except KeyError as e:
        print(f"\n✗ Error: Required column not found - {e}")
        print(f"Expected columns: {CONTENT_COLUMN}, {LABEL_COLUMN}, {CATEGORY_COLUMN}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

if __name__ == "__main__":
    test_data_loading()
