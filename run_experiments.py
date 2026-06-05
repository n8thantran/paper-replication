"""
Run all experiments for AdaCD paper replication.
Prioritizes smaller datasets and runs subsets of large ones.
"""

import json
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adacd import adacd_generate, default_generate

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    "xstest_safe": "data/xstest_safe.json",
    "xstest_unsafe": "data/xstest_unsafe.json",
    "orbench_hard": "data/orbench_hard.json",
    "oktest": "data/oktest.json",
    "advbench": "data/advbench.json",
    "jailbench": "data/jailbench.json",
}

# Max samples per dataset (None = all)
MAX_SAMPLES = {
    "xstest_safe": None,     # 250 - small, run all
    "xstest_unsafe": None,   # 200 - small, run all
    "orbench_hard": 300,     # 1319 total, sample 300 for time
    "oktest": None,          # 250/300 - small, run all
    "advbench": 200,         # 520 total, sample 200 for time
    "jailbench": None,       # 100 - small, run all
}


def run_single(model, tokenizer, dataset_name, method, max_samples=None):
    """Run a single experiment with resumption."""
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{method}.json")
    
    with open(DATASETS[dataset_name]) as f:
        data = json.load(f)
    
    if max_samples and max_samples < len(data):
        data = data[:max_samples]
    total = len(data)
    
    # Check existing
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        if len(results) >= total:
            print(f"  SKIP {dataset_name}_{method}: already complete ({len(results)}/{total})")
            return results
        print(f"  RESUME {dataset_name}_{method}: {len(results)}/{total}")
    
    start_idx = len(results)
    print(f"\n{'='*60}")
    print(f"  {method.upper()} on {dataset_name} ({start_idx}/{total})")
    print(f"{'='*60}")
    
    t_start = time.time()
    
    for i in range(start_idx, total):
        prompt = data[i]["prompt"]
        
        t0 = time.time()
        try:
            if method == "default":
                response = default_generate(model, tokenizer, prompt, max_new_tokens=512)
            else:
                response = adacd_generate(model, tokenizer, prompt,
                                         alpha=4.5, lambda_=0.9, beta=0.01,
                                         k=10, max_new_tokens=512)
        except Exception as e:
            print(f"  ERROR sample {i}: {e}")
            response = f"[ERROR: {str(e)}]"
        
        elapsed = time.time() - t0
        results.append({
            "index": i,
            "prompt": prompt,
            "response": response,
            "method": method,
            "time": elapsed,
        })
        
        if (i + 1) % 10 == 0 or i == total - 1:
            done = i - start_idx + 1
            avg = (time.time() - t_start) / done
            eta = avg * (total - i - 1) / 60
            print(f"  [{i+1}/{total}] {elapsed:.1f}s avg={avg:.1f}s ETA={eta:.0f}min")
        
        if (i + 1) % 25 == 0 or i == total - 1:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - t_start
    print(f"  Done: {dataset_name}_{method} in {total_time/60:.1f}min")
    return results


def main():
    # Parse optional args
    only_method = sys.argv[1] if len(sys.argv) > 1 else "both"
    only_dataset = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("Loading Qwen3-8B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    print("Model loaded!")
    
    methods = ["default", "adacd"] if only_method == "both" else [only_method]
    datasets = [only_dataset] if only_dataset else list(DATASETS.keys())
    
    # Run in priority order: small datasets first
    priority = ["xstest_safe", "xstest_unsafe", "oktest", "jailbench", "advbench", "orbench_hard"]
    datasets = [d for d in priority if d in datasets]
    
    for method in methods:
        for ds in datasets:
            run_single(model, tokenizer, ds, method, MAX_SAMPLES.get(ds))
    
    print("\n\nAll experiments complete!")


if __name__ == "__main__":
    main()
