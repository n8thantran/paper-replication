"""
Run all experiments for AdaCD paper replication.
Generates outputs for all 6 datasets with both Default and AdaCD methods.
"""

import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adacd import adacd_generate, default_generate

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All experiments to run: (dataset_name, method)
EXPERIMENTS = [
    # Default baselines (remaining)
    ("orbench_hard", "default"),
    ("advbench", "default"),
    ("jailbench", "default"),
    # AdaCD on all datasets
    ("xstest_safe", "adacd"),
    ("xstest_unsafe", "adacd"),
    ("orbench_hard", "adacd"),
    ("oktest", "adacd"),
    ("advbench", "adacd"),
    ("jailbench", "adacd"),
]

DATASETS = {
    "xstest_safe": "data/xstest_safe.json",
    "xstest_unsafe": "data/xstest_unsafe.json",
    "orbench_hard": "data/orbench_hard.json",
    "oktest": "data/oktest.json",
    "advbench": "data/advbench.json",
    "jailbench": "data/jailbench.json",
}


def run_experiment(model, tokenizer, dataset_name, method, 
                   alpha=4.5, lambda_=0.9, beta=0.01, k=10, max_new_tokens=512):
    """Run a single experiment with resumption support."""
    
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{method}.json")
    
    # Load dataset
    with open(DATASETS[dataset_name]) as f:
        data = json.load(f)
    total = len(data)
    
    # Check for existing results
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        if len(results) >= total:
            print(f"  {dataset_name}_{method}: Already complete ({len(results)}/{total})")
            return results
        print(f"  Resuming from {len(results)}/{total}")
    
    start_idx = len(results)
    print(f"\n{'='*60}")
    print(f"Running {method} on {dataset_name} ({total} samples, starting from {start_idx})")
    print(f"{'='*60}")
    
    t_start = time.time()
    
    for i in range(start_idx, total):
        prompt = data[i]["prompt"]
        
        t0 = time.time()
        try:
            if method == "default":
                response = default_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            elif method == "adacd":
                response = adacd_generate(model, tokenizer, prompt,
                                         alpha=alpha, lambda_=lambda_, beta=beta,
                                         k=k, max_new_tokens=max_new_tokens)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            response = f"[ERROR: {str(e)}]"
        
        elapsed = time.time() - t0
        
        results.append({
            "index": i,
            "prompt": prompt,
            "response": response,
            "method": method,
            "time": elapsed,
        })
        
        # Print progress
        if (i + 1) % 10 == 0 or i == total - 1:
            avg_time = (time.time() - t_start) / (i - start_idx + 1)
            remaining = avg_time * (total - i - 1)
            print(f"  [{i+1}/{total}] ({elapsed:.1f}s, avg {avg_time:.1f}s, ETA {remaining/60:.1f}min)")
        
        # Save periodically
        if (i + 1) % 25 == 0 or i == total - 1:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - t_start
    print(f"  Completed {dataset_name}_{method} in {total_time/60:.1f} minutes")
    return results


def main():
    print("Loading Qwen3-8B model...")
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")
    
    for dataset_name, method in EXPERIMENTS:
        run_experiment(model, tokenizer, dataset_name, method)
    
    print("\n\nAll experiments complete!")
    print("Run 'python evaluate.py outputs/' to evaluate results.")


if __name__ == "__main__":
    main()
