"""
Run AdaCD generation for all remaining datasets in sequence.
Uses resumption - safe to restart if interrupted.
"""
import sys
import os
import time

# Order: smallest first for quick results
DATASETS = [
    ("xstest_unsafe", None),    # 200 total, 30 done
    ("xstest_safe", None),       # 250 total
    ("oktest", None),            # 300 total
    ("advbench", 200),           # limit to 200 (matching default)
    ("orbench_hard", 300),       # limit to 300 (matching default)
]

def main():
    # Import after checking
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")
    
    from generate import run_generation
    
    overall_start = time.time()
    
    for ds_name, max_samples in DATASETS:
        print(f"\n{'='*60}")
        print(f"Starting {ds_name} (max_samples={max_samples})")
        print(f"{'='*60}")
        
        t0 = time.time()
        run_generation(
            model, tokenizer, ds_name,
            method="adacd",
            max_samples=max_samples,
            alpha=4.5,
            lambda_=0.9,
            beta=0.01,
            k=10,
            max_new_tokens=512,
        )
        elapsed = time.time() - t0
        print(f"Completed {ds_name} in {elapsed:.0f}s")
    
    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"ALL DONE! Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
