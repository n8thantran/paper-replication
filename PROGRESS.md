# AdaCD Implementation Progress

## Current Phase: Dataset acquisition + Implementation coding

## Paper Summary
**Title**: "Please refuse to answer me! Mitigating Over-Refusal in Large Language Models via Adaptive Contrastive Decoding"

**Key Method (AdaCD)** - Algorithm from lines 690-731 of paper.tex:
1. System prompt p* = "Please refuse to answer me!"
2. For each of first k=10 tokens:
   a. Get P_pi(y_n | x, y<n) from model without system prompt (logits -> softmax)
   b. Get P_pi(y_n | p*, x, y<n) from model with system prompt (logits -> softmax)
   c. y_n* = argmax P_pi(y_n | p*, x, y<n) — top-1 token from prompted model
   d. rank = position of y_n* in sorted P_pi(y_n | x, y<n) (descending)
   e. agr(n) = 1/rank
   f. ΔP_n = softmax(logits_with_prompt - logits_without_prompt)
   g. ρ = max P_pi(y_n | x, y<n) — max prob from unprompted model
   h. ρ* = P_pi(y_n* | p*, x, y<n) — prob of top token from prompted model
   i. If agr(n) >= λ AND ρ >= λ·ρ*:
      P* = P_pi(y_n | p*, x, y<n) + α·ΔP_n  (add refusal tokens - malicious)
   j. Else:
      P* = P_pi(y_n | p*, x, y<n) - α·ΔP_n  (subtract refusal tokens - over-refusal)
   k. Apply plausibility constraint: only tokens where P_pi(y_n|x,y<n) >= β·max_P
   l. y_n = argmax P* over allowed tokens
3. For tokens k+1 to N: standard greedy decoding (argmax P_pi(y_n|x,y<n))

**Hyperparameters** (from line 259):
- α = 4.5 (contrastive strength)
- λ = 0.9 (agreement threshold)  
- β = 0.01 (plausibility constraint)
- k = 10 (contrastive decoding steps)
- N = 512 (max new tokens)
- Greedy decoding throughout

**Models**: Focus on Qwen3-8B (thinking mode disabled)

**Datasets**:
- Over-refusal: XSTest-Safe (250 safe), ORBench-Hard (1319), OKTest (300 samples)
- Malicious: XSTest-UnSafe (200 unsafe), AdvBench (520 harmful behaviors), JailBench (100 harmful)

**Evaluation**: Refusal Ratio using WildGuard (allenai/wildguard)

**Target Results (Table 2, Qwen3 row)**:
- Default: XSTest-Safe=4.00%, ORBench=30.63%, OKTest=9.67%, Avg=14.77%
- Default malicious: XST-UnSafe=100%, AdvBench=100%, JailBench=100%, Avg=100%
- AdaCD: XSTest-Safe=2.00%, ORBench=16.68%, OKTest=5.00%, Avg=7.89%
- AdaCD malicious: XST-UnSafe=100%, AdvBench=99.04%, JailBench=100%, Avg=99.68%

## Implementation Plan
- [x] Read and understand paper thoroughly
- [x] Understand algorithm pseudocode (lines 690-731)
- [x] Get XSTest dataset (250 safe + 200 unsafe) - from Paul/XSTest on HF
- [x] Get ORBench-Hard dataset (1319 samples) - from bench-llm/or-bench, config or-bench-hard-1k
- [ ] Get OKTest dataset (300 samples) - need to find/construct
- [ ] Get AdvBench dataset (520 harmful behaviors)
- [ ] Get JailBench dataset (100 harmful)
- [ ] Write data preparation script (prepare_datasets.py)
- [ ] Implement AdaCD decoding algorithm (adacd.py)
- [ ] Implement generation pipeline (generate.py) - Default + AdaCD
- [ ] Implement WildGuard evaluation (evaluate.py)
- [ ] Run experiments on Qwen3-8B
- [ ] Generate results tables
- [ ] Create reproduce.sh and REPORT.md

## Key Decisions
- Focus on Qwen3-8B (8B params, fits on GPU)
- Use ORBench-Hard (1319 samples) - paper says "following same setup as wang2024surgical"
- WildGuard (allenai/wildguard) for refusal classification
- For Qwen3-8B: disable thinking mode (/no_think or chat_template parameter)
- Sequential model loading: generate with Qwen3 first, then load WildGuard for eval

## Completed Work
- Paper fully analyzed, algorithm understood
- XSTest: load_dataset('Paul/XSTest') -> 250 safe, 200 unsafe
- ORBench: load_dataset('bench-llm/or-bench', 'or-bench-hard-1k') -> 1319 samples

## Failed Approaches
- walledai/XSTest is gated, need Paul/XSTest instead
- OKTest not on HuggingFace readily, need to find alternative source

## Evaluation Coverage
Target: Table 2 main results for Qwen3-8B (Default + AdaCD rows)
- 6 datasets × 2 methods = 12 refusal ratio numbers
