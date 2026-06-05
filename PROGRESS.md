# AdaCD Implementation Progress

## Current Phase: Planning and initial implementation

## Paper Summary
**Title**: "Please refuse to answer me! Mitigating Over-Refusal in Large Language Models via Adaptive Contrastive Decoding"

**Key Method (AdaCD)**:
1. Uses an extreme system prompt `p* = "Please refuse to answer me!"` to extract refusal token distribution
2. Computes ΔP_n = softmax(logits_with_prompt - logits_without_prompt) as refusal token distribution
3. Adaptive decoding mode switch based on:
   - Agreement ratio: agr(n) = 1/rank(y_n*) where y_n* is top-1 token from prompted model
   - Adaptive confidence constraint: ρ < λ·ρ*
4. Final decoding: P*(y_n|x) = P(y_n|p*,x) + α·I(n)·ΔP_n
   - I(n) = +1 if agr(n) >= λ AND ρ >= λ·ρ* (add refusal tokens for malicious)
   - I(n) = -1 otherwise (subtract refusal tokens for over-refusal)
5. Adaptive plausibility constraint: only select from tokens where P(y_n|x) >= β·max P
6. Applied only to first k=10 tokens, then greedy decoding

**Hyperparameters**:
- α = 4.5 (contrastive strength)
- λ = 0.9 (agreement threshold)
- β = 0.01 (plausibility constraint)
- k = 10 (number of contrastive decoding steps)
- N = 512 (max new tokens)
- Greedy decoding

**Models**:
- Llama3-8B-Instruct (meta-llama/Meta-Llama-3-8B-Instruct)
- Gemma2-9B-It (google/gemma-2-9b-it)
- Qwen3-8B (Qwen/Qwen3-8B) - thinking mode disabled

**Datasets**:
- Over-refusal: XSTest-Safe, ORBench, OKTest
- Malicious: XSTest-UnSafe, AdvBench, JailBench

**Evaluation**:
- Refusal Ratio using WildGuard (LLM-based safety evaluation)
- Key results in Table 2 (main results)

## Implementation Plan
- [x] Read and understand paper
- [ ] Set up environment and install dependencies
- [ ] Download/prepare datasets (XSTest, ORBench, OKTest, AdvBench, JailBench)
- [ ] Implement AdaCD decoding algorithm
- [ ] Implement evaluation pipeline with WildGuard
- [ ] Run experiments on Qwen3-8B (smallest feasible given GPU constraints)
- [ ] Generate results tables
- [ ] Create reproduce.sh and REPORT.md

## Key Decisions
- Will focus on Qwen3-8B first as it's the most feasible model
- Use WildGuard for evaluation as specified in paper
- Greedy decoding for default, AdaCD for first k tokens then greedy

## Completed Work
(none yet)

## Failed Approaches
(none yet)

## Evaluation Coverage
Target: Table 2 main results, at minimum for one model
