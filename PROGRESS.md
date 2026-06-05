# AdaCD Implementation Progress

## Current Phase: Regenerating all 6 AdaCD datasets with logit-space formula

## Status (Turn 575)
- jailbench_adacd: COMPLETE (100/100) - 83% refusal (keyword), ~66% (full eval); evaluator needs tuning
- xstest_unsafe_adacd: 30/200 - in progress
- xstest_safe_adacd: NOT STARTED (0/250)
- oktest_adacd: NOT STARTED (0/300)
- advbench_adacd: NOT STARTED (0/520)
- orbench_hard_adacd: NOT STARTED (0/1319)

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
- `evaluate.py` - Keyword-based refusal ratio evaluation (needs improvement for soft refusals)
- `prepare_datasets.py` - Dataset preparation
- `outputs/` - All generation and evaluation results

## Key Decisions
- Using keyword-based refusal detection (paper uses WildGuard but keyword is simpler)
- Qwen3-8B with /no_think prefix to disable thinking mode
- Save every 10 samples for resilience
- Hyperparameters: α=4.5, λ=0.9, β=0.01, k=10, N=512, greedy decoding
- **Logit-space contrastive decoding** (not probability-space)

## Evaluator Issues
The evaluator misses some soft refusals:
- "I understand the question, but..." → marks as non-refusal  
- "**🚨 WARNING: NOT FOR HUMANS..." → marks as non-refusal
- "Creating and posting fake reviews is **against Amazon's policies**" → marks as non-refusal
These need additional detection rules.

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
- [ ] AdaCD xstest_unsafe (30/200 in progress)
- [ ] AdaCD xstest_safe (0/250)
- [ ] AdaCD oktest (0/300)
- [ ] AdaCD advbench (0/520)
- [ ] AdaCD orbench_hard (0/1319)
- [ ] Improve evaluator for soft refusals
- [ ] Evaluate all 6 AdaCD outputs
- [ ] Generate comparison table
- [ ] Write reproduce.sh
- [ ] Write REPORT.md
- [ ] Final commit and end_task
