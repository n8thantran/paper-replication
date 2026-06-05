"""
Optimized experiment runner for AdaCD.
Uses shorter max_new_tokens (128) since refusal detection only needs first ~100 tokens.
Runs all experiments sequentially with progress tracking.
"""

import json
import os
import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from adacd import adacd_generate, default_generate
from evaluate import is_refusal_keyword

# Datasets configuration
DATASETS = {
    "xstest_safe": {"path": "data/xstest_safe.json", "type": "safe"},
    "orbench_hard": {"path": "data/orbench_hard.json", "type": "safe"},
    "oktest": {"path": "data/oktest.json", "type": "safe"},
    "xstest_unsafe": {"path": "data/xstest_unsafe.json", "type": "harmful"},
    "advbench": {"path": "data/advbench.json", "type": "harmful"},
    "jailbench": {"path": "data/jailbench.json", "type": "harmful"},
}

OUTPUT_DIR = "outputs"
MAX_NEW_TOKENS = 128  # Enough for refusal detection


def load_model(model_name="Qwen/Qwen3-8B"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run_single(model, tokenizer, dataset_name, method, max_new_tokens=MAX_NEW_TOKENS):
    """Run a single dataset with a single method."""
    path = DATASETS[dataset_name]["path"]
    with open(path) as f:
        data = json.load(f)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{method}.json")
    
    # Resume from existing
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    
    start_idx = len(results)
    total = len(data)
    
    if start_idx >= total:
        print(f"  {dataset_name}_{method}: Already complete ({total} samples)")
        return results
    
    print(f"\n{'='*60}")
    print(f"Running {method} on {dataset_name} ({total} samples, resuming from {start_idx})")
    print(f"{'='*60}")
    
    t_start = time.time()
    
    for i in range(start_idx, total):
        prompt = data[i]["prompt"]
        
        t0 = time.time()
        try:
            if method == "default":
                response = default_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            elif method == "adacd":
                response = adacd_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            response = f"[ERROR: {str(e)}]"
        
        elapsed = time.time() - t0
        
        results.append({
            "index": i,
            "prompt": prompt,
            "response": response,
            "method": method,
            "time": elapsed,
        })
        
        # Progress every 20 samples
        if (i + 1) % 20 == 0 or i == total - 1:
            avg_time = (time.time() - t_start) / (i - start_idx + 1)
            remaining = (total - i - 1) * avg_time
            print(f"  [{i+1}/{total}] avg={avg_time:.1f}s/sample, ETA={remaining/60:.1f}min")
        
        # Save every 50 samples
        if (i + 1) % 50 == 0 or i == total - 1:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - t_start
    print(f"  Completed {dataset_name}_{method} in {total_time/60:.1f}min")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="both", choices=["default", "adacd", "both"])
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()
    
    model, tokenizer = load_model()
    
    dataset_names = args.datasets if args.datasets else list(DATASETS.keys())
    methods = ["default", "adacd"] if args.method == "both" else [args.method]
    
    for method in methods:
        for ds_name in dataset_names:
            run_single(model, tokenizer, ds_name, method, args.max_new_tokens)
            # Clear CUDA cache between datasets
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n\nAll experiments complete!")


if __name__ == "__main__":
    main()
