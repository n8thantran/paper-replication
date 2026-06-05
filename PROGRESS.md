# AdaCD Implementation Progress

## Current Phase: AdaCD generation nearly complete (5/6 datasets done, advbench in progress)

## Status (Turn 650)
- jailbench_adacd: COMPLETE (100/100) ✅
- xstest_unsafe_adacd: COMPLETE (200/200) ✅
- xstest_safe_adacd: COMPLETE (250/250) ✅
- oktest_adacd: COMPLETE (300/300) ✅
- advbench_adacd: IN PROGRESS (~50/200, ~2s/sample)
- orbench_hard_adacd: QUEUED (300 samples, ~11s/sample → ~55min)

Background process PID 105958 running `run_all_adacd.py` → `adacd_generation_log.txt`

## Estimated Time Remaining
- advbench: ~5 min remaining
- orbench_hard: ~55 min (300 samples × ~11s)
- Total: ~60 min

## Algorithm Implementation
Using **logit-space** contrastive decoding:
- Over-refusal: `logits* = logits_p - α*(logits_p - logits_u)`
- Malicious: `logits* = logits_p + α*(logits_p - logits_u)`
- Agreement ratio check: agr >= λ AND ρ >= λ*ρ* → malicious; else → over-refusal
- Plausibility constraint: P_u >= β * max(P_u)

## Paper Target Results (Table 2, Qwen3 row)
| Dataset | Default RR | AdaCD RR |
|---------|-----------|----------|
| XSTest-Safe | 4.00% | 2.00% |
| ORBench-Hard | 30.63% | 16.68% |
| OKTest | 9.67% | 5.00% |
| XSTest-Unsafe | 100% | 100% |
| AdvBench | 100% | 99.04% |
| JailBench | 100% | 100% |

## Our Default Results (keyword-based eval) - VERIFIED GOOD
| Dataset | Our RR | Paper RR |
|---------|--------|----------|
| XSTest-Safe | 5.20% | 4.00% |
| ORBench-Hard | 29.87% | 30.63% |
| OKTest | 9.33% | 9.67% |
| XSTest-Unsafe | 100.00% | 100% |
| AdvBench | 100.00% | 100% |
| JailBench | 100.00% | 100% |

## Key Files
- `adacd.py` - Core AdaCD algorithm (logit-space, committed)
- `generate.py` - Generation pipeline with resumption support
- `evaluate.py` - Keyword-based refusal ratio evaluation
- `evaluate_all.py` - Runs evaluation on all datasets, produces summary table
- `prepare_datasets.py` - Dataset preparation
- `run_all_adacd.py` - Batch AdaCD generation script (currently running)
- `outputs/` - All generation and evaluation results
- `reproduce.sh` - Reproduction script

## Key Decisions
- Using keyword-based refusal detection (paper uses WildGuard but keyword is simpler)
- Qwen3-8B with /no_think prefix to disable thinking mode
- Save every 10 samples for resilience
- Hyperparameters: α=4.5, λ=0.9, β=0.01, k=10, N=512, greedy decoding
- **Logit-space contrastive decoding** (not probability-space)

## What's Left After Generation Completes
1. Run evaluate_all.py to get final numbers
2. Copy results to /workspace/results/
3. Create REPORT.md
4. Verify reproduce.sh works (at least evaluation part)
5. Final commit and push

## Failed Approaches
1. **Probability-space formula**: `P* = P_prompted - α·softmax(logit_diff)` - doesn't work because
   softmax distributes ΔP across too many tokens, making individual corrections too small.
   The logit-space version `logits* = logits_p - α*(logits_p - logits_u)` works correctly.

## Implementation Plan
- [x] Dataset preparation (prepare_datasets.py)
- [x] AdaCD algorithm (adacd.py) - logit-space implementation
- [x] Generation pipeline (generate.py)
- [x] Evaluation (evaluate.py)
- [x] Default generation for all 6 datasets
- [x] Default evaluation for all 6 datasets
- [x] AdaCD jailbench (100/100 DONE)
- [x] AdaCD xstest_unsafe (200/200 DONE)
- [x] AdaCD xstest_safe (250/250 DONE)
- [x] AdaCD oktest (300/300 DONE)
- [ ] AdaCD advbench (~50/200 in progress)
- [ ] AdaCD orbench_hard (0/300 queued)
- [ ] Run evaluate_all.py
- [ ] Copy results to /workspace/results/
- [ ] Create REPORT.md
- [ ] Verify reproduce.sh
- [ ] Final commit
