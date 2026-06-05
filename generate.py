"""
Generation pipeline for AdaCD experiments.
Runs Default and AdaCD generation on all 6 datasets using Qwen3-8B.
"""

import json
import os
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adacd import adacd_generate, default_generate

# Datasets configuration
DATASETS = {
    # Over-refusal datasets (safe queries)
    "xstest_safe": {"path": "data/xstest_safe.json", "type": "safe"},
    "orbench_hard": {"path": "data/orbench_hard.json", "type": "safe"},
    "oktest": {"path": "data/oktest.json", "type": "safe"},
    # Malicious datasets (harmful queries)
    "xstest_unsafe": {"path": "data/xstest_unsafe.json", "type": "harmful"},
    "advbench": {"path": "data/advbench.json", "type": "harmful"},
    "jailbench": {"path": "data/jailbench.json", "type": "harmful"},
}

OUTPUT_DIR = "outputs"


def load_model(model_name: str = "Qwen/Qwen3-8B", device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer


def load_dataset(dataset_name: str):
    """Load a dataset from JSON file."""
    path = DATASETS[dataset_name]["path"]
    with open(path, "r") as f:
        data = json.load(f)
    return data


def run_generation(
    model,
    tokenizer,
    dataset_name: str,
    method: str = "default",
    max_samples: int = None,
    alpha: float = 4.5,
    lambda_: float = 0.9,
    beta: float = 0.01,
    k: int = 10,
    max_new_tokens: int = 512,
):
    """
    Run generation on a dataset with specified method.
    
    Args:
        method: "default" or "adacd"
    """
    data = load_dataset(dataset_name)
    if max_samples is not None:
        data = data[:max_samples]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{method}.json")
    
    # Check for existing results to enable resumption
    existing_results = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_results = json.load(f)
        print(f"  Found {len(existing_results)} existing results, resuming...")
    
    results = existing_results
    start_idx = len(results)
    
    total = len(data)
    print(f"\nRunning {method} on {dataset_name} ({total} samples, starting from {start_idx})")
    
    for i in range(start_idx, total):
        prompt = data[i]["prompt"]
        
        t0 = time.time()
        try:
            if method == "default":
                response = default_generate(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                )
            elif method == "adacd":
                response = adacd_generate(
                    model, tokenizer, prompt,
                    alpha=alpha,
                    lambda_=lambda_,
                    beta=beta,
                    k=k,
                    max_new_tokens=max_new_tokens,
                )
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
            print(f"  [{i+1}/{total}] ({elapsed:.1f}s) {prompt[:60]}...")
        
        # Save periodically
        if (i + 1) % 50 == 0 or i == total - 1:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(results)} results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run AdaCD generation experiments")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--method", type=str, default="both", choices=["default", "adacd", "both"],
                        help="Generation method")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Datasets to run on (default: all)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per dataset (for testing)")
    parser.add_argument("--alpha", type=float, default=4.5)
    parser.add_argument("--lambda_", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Determine datasets
    dataset_names = args.datasets if args.datasets else list(DATASETS.keys())
    
    # Determine methods
    methods = ["default", "adacd"] if args.method == "both" else [args.method]
    
    # Run experiments
    for method in methods:
        for ds_name in dataset_names:
            run_generation(
                model, tokenizer, ds_name,
                method=method,
                max_samples=args.max_samples,
                alpha=args.alpha,
                lambda_=args.lambda_,
                beta=args.beta,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
            )
    
    print("\nAll generation complete!")


if __name__ == "__main__":
    main()
