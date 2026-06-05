"""
Batch AdaCD generation for all 6 datasets.
Processes datasets in order from smallest to largest.
Saves progress every 10 samples for resilience.
"""

import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adacd import adacd_generate

# Dataset order: smallest first for quick verification
DATASETS = [
    ("jailbench", "data/jailbench.json"),
    ("xstest_unsafe", "data/xstest_unsafe.json"),
    ("xstest_safe", "data/xstest_safe.json"),
    ("oktest", "data/oktest.json"),
    ("advbench", "data/advbench.json"),
    ("orbench_hard", "data/orbench_hard.json"),
]

SAVE_EVERY = 10

def load_progress(output_path):
    """Load existing results for resumption."""
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return []

def save_results(results, output_path):
    """Save results to file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    print("Loading model...")
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print("Model loaded!")
    
    for dataset_name, data_path in DATASETS:
        output_path = f"outputs/{dataset_name}_adacd.json"
        
        # Load dataset
        with open(data_path) as f:
            dataset = json.load(f)
        
        # Load existing progress
        results = load_progress(output_path)
        start_idx = len(results)
        total = len(dataset)
        
        if start_idx >= total:
            print(f"\n=== {dataset_name}: Already complete ({total}/{total}) ===")
            continue
        
        print(f"\n=== {dataset_name}: Resuming from {start_idx}/{total} ===")
        t0 = time.time()
        
        for i in range(start_idx, total):
            item = dataset[i]
            prompt = item["prompt"]
            
            try:
                response = adacd_generate(
                    model, tokenizer, prompt,
                    alpha=4.5, lambda_=0.9, beta=0.01,
                    k=10, max_new_tokens=512
                )
            except Exception as e:
                print(f"  ERROR on sample {i}: {e}")
                response = f"[ERROR: {str(e)}]"
            
            result = {
                "id": item.get("id", i),
                "prompt": prompt,
                "response": response,
                "method": "adacd",
                "label": item.get("label", ""),
            }
            if "category" in item:
                result["category"] = item["category"]
            
            results.append(result)
            
            # Save periodically
            if (i + 1) % SAVE_EVERY == 0 or (i + 1) == total:
                save_results(results, output_path)
                elapsed = time.time() - t0
                rate = (i - start_idx + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{dataset_name}] {i+1}/{total} ({rate:.2f} samples/s, ~{remaining/60:.1f}min remaining)")
        
        elapsed = time.time() - t0
        print(f"  {dataset_name} complete! {total} samples in {elapsed/60:.1f}min")

if __name__ == "__main__":
    main()
