#!/usr/bin/env python3
"""
Run AdaCD generation on all 6 datasets with checkpoint/resume support.
Processes datasets in priority order: smallest first for fast results.
"""

import json
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adacd import adacd_generate

MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"

DATASETS = [
    # (name, max_samples) - ordered by priority/size  
    ("jailbench", None),      # 100
    ("xstest_unsafe", None),  # 200
    ("xstest_safe", None),    # 250
    ("oktest", None),         # 300
    ("advbench", None),       # 520
    ("orbench_hard", None),   # 1319
]


def load_dataset(name):
    with open(f"{DATA_DIR}/{name}.json") as f:
        return json.load(f)


def save_output(name, results):
    path = f"{OUTPUT_DIR}/{name}_adacd.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_existing(name):
    path = f"{OUTPUT_DIR}/{name}_adacd.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def main():
    # Filter to specific dataset if given as argument
    filter_ds = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    for ds_name, max_samples in DATASETS:
        if filter_ds and ds_name != filter_ds:
            continue
            
        data = load_dataset(ds_name)
        if max_samples:
            data = data[:max_samples]
        
        # Resume from checkpoint
        results = load_existing(ds_name)
        done = len(results)
        total = len(data)
        
        if done >= total:
            print(f"\n=== {ds_name}: Already complete ({done}/{total}) ===")
            continue
        
        print(f"\n=== {ds_name}: Resuming from {done}/{total} ===")
        
        start_time = time.time()
        for i in range(done, total):
            item = data[i]
            query = item.get("query", item.get("prompt", ""))
            
            try:
                response = adacd_generate(model, tokenizer, query)
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                response = f"[ERROR] {str(e)}"
            
            result = {
                "id": item.get("id", i),
                "query": query,
                "response": response,
                "method": "adacd",
            }
            if "category" in item:
                result["category"] = item["category"]
            
            results.append(result)
            
            # Save every 10 samples
            if (i + 1) % 10 == 0 or i == total - 1:
                save_output(ds_name, results)
                elapsed = time.time() - start_time
                rate = (i - done + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  {ds_name}: {i+1}/{total} ({rate:.2f} samples/s, ETA: {eta/60:.1f}min)")
        
        elapsed = time.time() - start_time
        print(f"  {ds_name}: Complete! {total} samples in {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
