"""
Test script to run the pipeline with both zero-shot and few-shot prompting strategies.
This will process the sample data and show the augmentation results.
"""
import sys
import os

# Modify the data-pipeline.py to run both strategies
def run_both_strategies():
    print("\n" + "="*80)
    print("RUNNING PIPELINE WITH BOTH PROMPTING STRATEGIES")
    print("="*80)
    
    # Import after setting the path
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import reload
    import data_pipeline as dp
    
    strategies = ["zero-shot", "few-shot"]
    
    for strategy in strategies:
        print(f"\n\n{'='*80}")
        print(f"TESTING WITH {strategy.upper()} PROMPTING")
        print(f"{'='*80}\n")
        
        # Reload the module to reset configuration
        reload(dp)
        
        # Update the prompting strategy
        dp.Config.PROMPTING_STRATEGY = strategy
        
        # Run the pipeline
        try:
            dp.run_pipeline()
            print(f"\n✓ {strategy.upper()} PROMPTING COMPLETED SUCCESSFULLY")
        except Exception as e:
            print(f"\n✗ {strategy.upper()} PROMPTING FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"END OF {strategy.upper()} PROMPTING TEST")
        print(f"{'='*80}")

if __name__ == "__main__":
    run_both_strategies()
