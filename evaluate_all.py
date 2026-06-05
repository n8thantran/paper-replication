"""
Evaluate all datasets (default and adacd) and produce summary table.
"""
import json
import os
import sys

sys.path.insert(0, '/workspace')
from evaluate import is_refusal_keyword

DATASETS_INFO = {
    # Over-refusal datasets (lower RR is better)
    "xstest_safe": {"type": "safe", "paper_default": 4.00, "paper_adacd": 2.00},
    "orbench_hard": {"type": "safe", "paper_default": 30.63, "paper_adacd": 16.68},
    "oktest": {"type": "safe", "paper_default": 9.67, "paper_adacd": 5.00},
    # Malicious datasets (higher RR is better)
    "xstest_unsafe": {"type": "harmful", "paper_default": 100.0, "paper_adacd": 100.0},
    "advbench": {"type": "harmful", "paper_default": 100.0, "paper_adacd": 99.04},
    "jailbench": {"type": "harmful", "paper_default": 100.0, "paper_adacd": 100.0},
}


def evaluate_file(path):
    """Evaluate a generation output file and return refusal rate."""
    with open(path) as f:
        data = json.load(f)
    
    refusals = 0
    total = len(data)
    results = []
    
    for item in data:
        resp = item.get("response", "")
        is_ref = is_refusal_keyword(resp)
        if is_ref:
            refusals += 1
        results.append({**item, "is_refusal": is_ref})
    
    rr = (refusals / total * 100) if total > 0 else 0
    return rr, total, refusals, results


def main():
    os.makedirs("results", exist_ok=True)
    
    print("=" * 80)
    print("AdaCD Evaluation Results")
    print("=" * 80)
    
    summary = {}
    
    # Header
    print(f"\n{'Dataset':<20} {'Type':<8} {'Method':<10} {'N':>5} {'Refusals':>8} {'RR%':>8} {'Paper%':>8}")
    print("-" * 80)
    
    for ds_name, info in DATASETS_INFO.items():
        for method in ["default", "adacd"]:
            path = f"outputs/{ds_name}_{method}.json"
            if not os.path.exists(path):
                print(f"{ds_name:<20} {info['type']:<8} {method:<10} {'N/A':>5} {'N/A':>8} {'N/A':>8} {info[f'paper_{method}']:>8.2f}")
                continue
            
            rr, total, refusals, results = evaluate_file(path)
            paper_rr = info[f"paper_{method}"]
            
            # Save detailed eval results
            eval_path = f"outputs/eval_{ds_name}_{method}.json"
            with open(eval_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"{ds_name:<20} {info['type']:<8} {method:<10} {total:>5} {refusals:>8} {rr:>8.2f} {paper_rr:>8.2f}")
            
            summary[f"{ds_name}_{method}"] = {
                "dataset": ds_name,
                "method": method,
                "type": info["type"],
                "total": total,
                "refusals": refusals,
                "refusal_rate": round(rr, 2),
                "paper_refusal_rate": paper_rr,
            }
    
    # Save summary
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("Table 2 Comparison (Qwen3-8B)")
    print("=" * 80)
    print(f"\n{'Dataset':<20} {'Default (Ours)':>14} {'Default (Paper)':>15} {'AdaCD (Ours)':>13} {'AdaCD (Paper)':>14}")
    print("-" * 80)
    
    for ds_name, info in DATASETS_INFO.items():
        our_def = summary.get(f"{ds_name}_default", {}).get("refusal_rate", "N/A")
        our_ada = summary.get(f"{ds_name}_adacd", {}).get("refusal_rate", "N/A")
        paper_def = info["paper_default"]
        paper_ada = info["paper_adacd"]
        
        our_def_str = f"{our_def:.2f}%" if isinstance(our_def, (int, float)) else our_def
        our_ada_str = f"{our_ada:.2f}%" if isinstance(our_ada, (int, float)) else our_ada
        
        print(f"{ds_name:<20} {our_def_str:>14} {paper_def:>14.2f}% {our_ada_str:>13} {paper_ada:>13.2f}%")
    
    print()
    
    # Save formatted table
    with open("results/table2_comparison.txt", "w") as f:
        f.write("Table 2 Comparison: AdaCD Refusal Rate Results (Qwen3-8B)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Dataset':<20} {'Default (Ours)':>14} {'Default (Paper)':>15} {'AdaCD (Ours)':>13} {'AdaCD (Paper)':>14}\n")
        f.write("-" * 80 + "\n")
        for ds_name, info in DATASETS_INFO.items():
            our_def = summary.get(f"{ds_name}_default", {}).get("refusal_rate", "N/A")
            our_ada = summary.get(f"{ds_name}_adacd", {}).get("refusal_rate", "N/A")
            paper_def = info["paper_default"]
            paper_ada = info["paper_adacd"]
            our_def_str = f"{our_def:.2f}%" if isinstance(our_def, (int, float)) else our_def
            our_ada_str = f"{our_ada:.2f}%" if isinstance(our_ada, (int, float)) else our_ada
            f.write(f"{ds_name:<20} {our_def_str:>14} {paper_def:>14.2f}% {our_ada_str:>13} {paper_ada:>13.2f}%\n")
    
    print("Results saved to results/summary.json and results/table2_comparison.txt")


if __name__ == "__main__":
    main()
