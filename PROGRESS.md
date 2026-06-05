# AdaCD Implementation Progress

## Current Phase: Starting implementation - datasets + AdaCD algorithm

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

**Models**: Llama3-8B-Instruct, Gemma2-9B-It, Qwen3-8B (thinking mode disabled)

**Datasets**:
- Over-refusal: XSTest-Safe (250 safe), ORBench, OKTest (300 samples)
- Malicious: XSTest-UnSafe (200 unsafe), AdvBench (520 harmful behaviors), JailBench (100 harmful)

**Evaluation**: Refusal Ratio using WildGuard (allenai/wildguard)

**Target Results (Table 2, Qwen3 row)**:
- Over-refusal: XSTest-Safe=2.00%, ORBench=16.68%, OKTest=5.00%, Avg=7.89%
- Malicious: XSTest-UnSafe=100.00%, AdvBench=99.04%, JailBench=100.00%, Avg=99.68%
- Default baseline: XSTest-Safe=4.00%, ORBench=30.63%, OKTest=9.67%, Avg=14.77%

## Implementation Plan
- [x] Read and understand paper thoroughly
- [x] Understand algorithm pseudocode (lines 690-731)
- [ ] Set up environment and install dependencies
- [ ] Download/prepare datasets (XSTest, ORBench, OKTest, AdvBench, JailBench)
- [ ] Implement AdaCD decoding algorithm
- [ ] Implement Default baseline (just greedy decoding)
- [ ] Implement evaluation pipeline with WildGuard
- [ ] Run Default + AdaCD on Qwen3-8B across all 6 datasets
- [ ] Generate results tables comparing to paper
- [ ] Create reproduce.sh and REPORT.md

## Key Decisions
- Focus on Qwen3-8B (8B params, fits on H100 80GB along with WildGuard 7B)
- Will need to load model and WildGuard separately (or sequentially) due to memory
- Use HuggingFace datasets library for XSTest and ORBench
- For OKTest, may need to find/create from paper description
- WildGuard is based on Mistral-7B, ~14GB in fp16

## Completed Work
- Paper fully analyzed, algorithm understood

## Failed Approaches
(none yet)

## Evaluation Coverage
Target: Table 2 main results for Qwen3-8B (Default + AdaCD rows)
- 6 datasets × 2 methods = 12 refusal ratio numbers
