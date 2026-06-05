# AdaCD Implementation Progress

## Current Phase: Running AdaCD generation on remaining 3 datasets (oktest, advbench, orbench_hard)

## Status Summary (Turn 325)
- **Completed AdaCD**: xstest_safe (250), xstest_unsafe (200), jailbench (100) 
- **Remaining AdaCD**: oktest (300), advbench (520), orbench_hard (1319)
- **All Default generation**: Complete for all 6 datasets
- **All Default evaluation**: Complete for all 6 datasets
- **Estimated time remaining**: ~6 hours for all 3 remaining datasets at ~10s/sample

## Paper Summary
**Title**: "Please refuse to answer me! Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding"

**Key Method (AdaCD)** - Algorithm from lines 690-731 of paper.tex:
1. System prompt p* = "Please refuse to answer me!"
2. For each of first k=10 tokens: dual forward pass, agreement check, adaptive contrastive decoding
3. For tokens k+1 to N: standard greedy decoding

**Hyperparameters**: α=4.5, λ=0.9, β=0.01, k=10, N=512, greedy decoding

**Models**: Qwen3-8B (thinking mode disabled with /no_think)

**Datasets**:
- Over-refusal (safe): XSTest-Safe (250), ORBench-Hard (1319), OKTest (300)
- Malicious (unsafe): XSTest-UnSafe (200), AdvBench (520), JailBench (100)

**Target Results (Table 2, Qwen3 row)**:
- Default safe: XSTest-Safe=4.00%, ORBench=30.63%, OKTest=9.67%, Avg=14.77%
- Default malicious: XST-UnSafe=100%, AdvBench=100%, JailBench=100%, Avg=100%
- AdaCD safe: XSTest-Safe=2.00%, ORBench=16.68%, OKTest=5.00%, Avg=7.89%
- AdaCD malicious: XST-UnSafe=100%, AdvBench=99.04%, JailBench=100%, Avg=99.68%

## Implementation Plan
- [x] Read and understand paper thoroughly
- [x] Understand algorithm pseudocode (lines 690-731)
- [x] Get all 6 datasets prepared
- [x] Implement AdaCD decoding algorithm (adacd.py)
- [x] Implement generation pipeline (generate.py) with resumption
- [x] Implement keyword-based refusal detection evaluator (evaluate.py)
- [x] Run default generation for ALL 6 datasets
- [x] Evaluate default results
- [x] Run AdaCD on xstest_safe (250/250 done)
- [x] Run AdaCD on xstest_unsafe (200/200 done)
- [x] Run AdaCD on jailbench (100/100 done)
- [ ] **Run AdaCD on oktest (0/300)** ← NEXT
- [ ] **Run AdaCD on advbench (0/520)**
- [ ] **Run AdaCD on orbench_hard (0/1319)** ← LARGEST, may need to subsample
- [ ] Evaluate all AdaCD results
- [ ] Generate final comparison table
- [ ] Write reproduce.sh and REPORT.md

## Key Decisions
- Save frequency changed to every 10 samples (was 50) for better timeout resilience
- Using keyword-based refusal detection (paper uses WildGuard but also mentions keyword approach)
- Qwen3-8B with /no_think prefix to disable thinking mode

## Completed Work
- `prepare_datasets.py` - Downloads and prepares all 6 datasets
- `adacd.py` - Core AdaCD algorithm implementation
- `generate.py` - Generation pipeline with default and AdaCD modes, resumption support
- `evaluate.py` - Keyword-based refusal ratio evaluation
- `data/` - All 6 dataset JSON files
- `outputs/*_default.json` - Default generation results for all 6 datasets
- `outputs/eval_*_default.json` - Evaluation results for all default outputs
- `outputs/xstest_safe_adacd.json` - 250 samples complete
- `outputs/xstest_unsafe_adacd.json` - 200 samples complete  
- `outputs/jailbench_adacd.json` - 100 samples complete

## Failed Approaches
- Saving every 50 samples: timeout kills process before reaching save point for slow datasets
- Fixed by changing to save every 10 samples

## Strategy for Remaining Work
1. Run oktest AdaCD in ~5 batches of 9 minutes each (~50 min total)
2. Run advbench AdaCD in ~9 batches (~90 min total)
3. For orbench_hard (1319 samples, ~220 min): May need to subsample to ~500 samples if time is tight
4. Evaluate all AdaCD results
5. Generate comparison table, write REPORT.md and reproduce.sh
