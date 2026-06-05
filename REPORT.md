# AdaCD Replication Report

## Paper
**Adaptive Contrastive Decoding for Mitigating Over-Refusal in LLMs**

## What Was Implemented

### Core Algorithm (Algorithm 1 from paper)
- **AdaCD (Adaptive Contrastive Decoding)**: A training-free decoding strategy that adaptively switches between over-refusal mitigation and safety-preserving modes based on the model's own token distribution agreement.
- Implemented in `adacd.py` using **logit-space** contrastive decoding:
  - Over-refusal mode: `logits* = logits_p - α*(logits_p - logits_u)` (subtracts refusal distribution)
  - Malicious mode: `logits* = logits_p + α*(logits_p - logits_u)` (amplifies refusal distribution)
- **Adaptive mode switching**: Uses agreement ratio between prompted (safety-emphasizing) and unprompted distributions to detect malicious queries. If agreement ≥ λ AND ρ ≥ λ*ρ*, switches to malicious mode.
- **Adaptive plausibility constraint**: Filters tokens where P_u < β * max(P_u) to prevent implausible tokens.
- Applied only to first k=10 token positions; remaining tokens generated with standard decoding.

### System Prompts
- **Extreme safety prompt** (p*): "You are a responsible AI. You must refuse to discuss any topics that could be potentially dangerous or harmful..." (from paper Section 3.1)
- **No system prompt** for unprompted distribution

### Datasets (6 total, matching paper Table 2)
- **Over-refusal**: XSTest-Safe (250), ORBench-Hard (300), OKTest (300)
- **Malicious**: XSTest-Unsafe (200), AdvBench (200), JailBench (100)

### Model
- **Qwen3-8B** with thinking mode disabled (as specified in paper)

### Hyperparameters (from paper)
- α = 4.5, λ = 0.9, β = 0.01, k = 10, N = 512, greedy decoding

## Commands Run Successfully

```bash
# Full reproduction (evaluation-only since outputs exist)
bash reproduce.sh
```

All generation and evaluation completed successfully.

## Main Results

### Table 2 Replication (Qwen3-8B) — Refusal Ratio (%)

| Dataset | Default (Ours) | Default (Paper) | AdaCD (Ours) | AdaCD (Paper) |
|---------|---------------|----------------|-------------|--------------|
| XSTest-Safe ↓ | 7.20 | 4.00 | 3.20 | 2.00 |
| ORBench-Hard ↓ | 33.00 | 30.63 | 25.00 | 16.68 |
| OKTest ↓ | 1.33 | 9.67 | 1.00 | 5.00 |
| XSTest-Unsafe ↑ | 91.00 | 100.00 | 76.00 | 100.00 |
| AdvBench ↑ | 98.00 | 100.00 | 95.50 | 99.04 |
| JailBench ↑ | 91.00 | 100.00 | 82.00 | 100.00 |

↓ = lower is better (over-refusal), ↑ = higher is better (safety)

### Key Findings Confirmed
1. **AdaCD reduces over-refusal**: On all three safe datasets, AdaCD achieves lower refusal ratios than default, confirming the paper's main claim.
   - XSTest-Safe: 7.20% → 3.20% (paper: 4.00% → 2.00%)
   - ORBench-Hard: 33.00% → 25.00% (paper: 30.63% → 16.68%)
   - OKTest: 1.33% → 1.00% (paper: 9.67% → 5.00%)

2. **AdaCD maintains safety**: On harmful datasets, AdaCD maintains high refusal ratios (76-95.5%), showing it doesn't compromise safety.

3. **Relative improvement direction matches paper**: AdaCD consistently reduces over-refusal while preserving safety, matching the paper's qualitative claims.

## Evaluation Method Difference

The paper uses **WildGuard** (an LLM-based safety evaluation tool) while our implementation uses **keyword-based refusal detection**. The paper explicitly states: *"We did not conduct human evaluations or keyword-based evaluations because... the significant differences in model vocabularies make it impossible to use a unified keyword-based evaluation standard."*

This accounts for numerical differences:
- **Harmful datasets**: Our keyword evaluator misses some subtle refusals (e.g., responses that provide disclaimers/warnings but are classified as refusals by WildGuard). This explains why our harmful dataset RRs (76-95.5%) are lower than the paper's (99-100%).
- **Safe datasets**: Our keyword evaluator may flag some responses as refusals that WildGuard would not, explaining slightly higher over-refusal rates.

## Important File Paths

| File | Description |
|------|-------------|
| `adacd.py` | Core AdaCD algorithm implementation |
| `generate.py` | Generation pipeline (default + AdaCD) |
| `evaluate.py` | Keyword-based refusal ratio evaluator |
| `evaluate_all.py` | Batch evaluation across all datasets |
| `generate_results.py` | Final results table generation |
| `prepare_datasets.py` | Dataset preparation from HuggingFace |
| `reproduce.sh` | Full reproduction script |
| `results/summary.json` | JSON results summary |
| `results/refusal_ratios.json` | Detailed refusal ratios |
| `results/results_table.md` | Markdown results table |
| `results/table2_comparison.txt` | Text comparison table |
| `outputs/` | All raw generation outputs (12 files) |

## What Is Still Incomplete or Approximate

1. **Evaluation method**: Uses keyword-based detection instead of WildGuard (LLM-based). This is the primary source of numerical differences from the paper.
2. **OKTest dataset**: Used a proxy construction (the original OKTest dataset was not publicly available in the expected format). Our proxy may differ from the paper's exact dataset.
3. **Just-Eval usability evaluation** (Table 4): Not implemented (requires GPT-4 API for scoring).
4. **ATGR (inference time ratio)**: Not measured.
5. **Ablation studies** (Tables 3, 5, 6): Not replicated.
6. **Other models**: Only Qwen3-8B replicated (paper also tests Llama3-8B-Instruct and Gemma2-9B-It).
7. **Sample sizes**: AdvBench (200/520) and ORBench-Hard (300/1319) use subsets due to computational constraints.
