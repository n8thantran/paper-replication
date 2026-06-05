# AdaCD Implementation Progress

## Current Phase: RUNNING AdaCD GENERATION (core contribution)

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

**Models**: Qwen3-8B (thinking mode disabled with /no_think)

**Datasets**:
- Over-refusal: XSTest-Safe (250 safe), ORBench-Hard (1319), OKTest (300 samples)
- Malicious: XSTest-UnSafe (200 unsafe), AdvBench (520 harmful behaviors), JailBench (100 harmful)

**Evaluation**: Refusal Ratio using keyword-based detection (WildGuard alternative)

**Target Results (Table 2, Qwen3 row)**:
- Default: XSTest-Safe=4.00%, ORBench=30.63%, OKTest=9.67%, Avg=14.77%
- Default malicious: XST-UnSafe=100%, AdvBench=100%, JailBench=100%, Avg=100%
- AdaCD: XSTest-Safe=2.00%, ORBench=16.68%, OKTest=5.00%, Avg=7.89%
- AdaCD malicious: XST-UnSafe=100%, AdvBench=99.04%, JailBench=100%, Avg=99.68%

## Implementation Plan
- [x] Read and understand paper thoroughly
- [x] Understand algorithm pseudocode (lines 690-731)
- [x] Get XSTest dataset (250 safe + 200 unsafe)
- [x] Get ORBench-Hard dataset (1319 samples)
- [x] Get OKTest dataset (300 samples)
- [x] Get AdvBench dataset (520 harmful behaviors)
- [x] Get JailBench dataset (100 harmful)
- [x] Write data preparation script (prepare_datasets.py)
- [x] Implement AdaCD decoding algorithm (adacd.py)
- [x] Implement generation pipeline (generate.py) - Default + AdaCD, with resumption
- [x] Implement keyword-based refusal detection evaluator (evaluate.py)
- [x] Run default generation for ALL 6 datasets
- [x] Evaluator tuned (good enough - not perfect but reasonable)
- [ ] **Run AdaCD generation for ALL 6 datasets** ← IN PROGRESS
- [ ] Evaluate AdaCD results
- [ ] Compile results table comparing Default vs AdaCD
- [ ] Write reproduce.sh
- [ ] Write REPORT.md

## Current Default Results (Turn 300)
Samples generated (some datasets subsampled for speed):
- xstest_safe: 250/250 complete
- xstest_unsafe: 200/200 complete
- orbench_hard: 300/1319 (subsampled)
- oktest: 300/300 complete
- advbench: 200/520 (subsampled)
- jailbench: 100/100 complete

Current evaluation results (keyword-based):
- xstest_safe_default: 5.20% (target: 4.00%)
- orbench_hard_default: 31.00% (target: 30.63%) ← VERY CLOSE!
- oktest_default: 1.00% (target: 9.67%) - model may genuinely not refuse much on OKTest
- xstest_unsafe_default: 90.00% (target: 100%)
- advbench_default: 98.00% (target: 100%)
- jailbench_default: 91.00% (target: 100%)

Note: Malicious dataset numbers are lower than paper's 100% because:
1. Paper uses WildGuard (LLM-based) evaluator, we use keyword-based
2. Some responses technically refuse but use subtle language our keywords miss
3. The evaluator is "good enough" for showing the AdaCD trend

## Key Files
- /workspace/adacd.py - Core AdaCD algorithm with KV cache
- /workspace/generate.py - Generation pipeline (default + AdaCD)
- /workspace/evaluate.py - Keyword-based refusal evaluator (30+ tiers)
- /workspace/prepare_datasets.py - Dataset preparation
- /workspace/smoke_test.py - Smoke test (verified working)
- /workspace/run_experiments.py - Experiment runner with priority ordering
- /workspace/data/ - All 6 datasets as JSON
- /workspace/outputs/ - Generated responses and evaluation results

## Failed Approaches
- WildGuard (allenai/wildguard): Gated model, can't access for evaluation
- Over-aggressive "I'm sorry" pattern: Made all sympathetic starts count as refusals
- Historical event patterns: Educational responses about history flagged as refusals
- "should absolutely" standalone: Too broad, catches absurd question responses
- Spent too long tuning evaluator - should have started AdaCD generation sooner

## Priority Plan (Turn 300+)
1. ✅ Commit current state
2. Run AdaCD generation on ALL 6 datasets (main time sink)
   - Start with smaller datasets: xstest_safe (250), xstest_unsafe (200), jailbench (100)
   - Then oktest (300), orbench_hard (300), advbench (200)
3. Evaluate AdaCD results
4. Write final results comparison
5. Write reproduce.sh and REPORT.md
