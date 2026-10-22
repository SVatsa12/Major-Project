import torch
import time

print("="*60)
print("GPU AVAILABILITY CHECK")
print("="*60)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No CUDA GPU detected - will use CPU only")

print("\n" + "="*60)
print("VARIANT D TIME ESTIMATE")
print("="*60)
print("Variant D: Fine-tuned InLegalBERT cross-encoder")
print("Configuration:")
print("  - 3 seeds × 5 outer folds = 15 training runs")
print("  - Max 10 epochs per run with early stopping")
print("  - Inner validation split from training folds")
print("  - Max length: 384 tokens")
print("  - Learning rate: 2e-5")
print("  - Weighted cross-entropy")

if torch.cuda.is_available():
    print("\nWith GPU:")
    print("  - Estimated time per run: ~10-15 minutes (with early stopping)")
    print("  - Total time estimate: ~3-4 hours for all 15 runs")
    print("  - Recommendation: Run overnight or during downtime")
else:
    print("\nWith CPU only:")
    print("  - Estimated time per run: ~2-4 hours")
    print("  - Total time estimate: ~30-60 hours for all 15 runs")
    print("  - Recommendation: NOT FEASIBLE - requires GPU")

print("\nWaiting for go-ahead before starting Variant D...")
