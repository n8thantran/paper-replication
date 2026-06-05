# AdaCD Implementation Progress

## Current Phase: Running AdaCD on orbench_hard (710/1319 done)

## Status Summary (Turn 375)
- **All 6 Default generations**: COMPLETE
- **All 6 Default evaluations**: COMPLETE  
- **AdaCD complete**: xstest_safe(250), xstest_unsafe(200), jailbench(100), oktest(300), advbench(520)
- **AdaCD in progress**: orbench_hard (710/1319) ← ~60 samples per 9-min window, ~10 more windows needed
- **After orbench_hard**: Run evaluate.py on all AdaCD outputs, generate comparison table, write reproduce.sh + REPORT.md

## Paper Target Results (Table 2, Qwen3 row)
| Dataset | Default RR | AdaCD RR |
|---------|-----------|----------|
| XSTest-Safe | 4.00% | 2.00% |
| ORBench-Hard | 30.63% | 16.68% |
| OKTest | 9.67% | 5.00% |
| XSTest-Unsafe | 100% | 100% |
| AdvBench | 100% | 99.04% |
| JailBench | 100% | 100% |

## Our Default Results (keyword-based eval)
| Dataset | Our RR | Paper RR |
|---------|--------|----------|
| XSTest-Safe | 5.20% | 4.00% |
| ORBench-Hard | 29.87% | 30.63% |
| OKTest | 9.33% | 9.67% |
| XSTest-Unsafe | 100.00% | 100% |
| AdvBench | 100.00% | 100% |
| JailBench | 100.00% | 100% |

## Key Files
- `adacd.py` - Core AdaCD algorithm (α=4.5, λ=0.9, β=0.01, k=10)
- `generate.py` - Generation pipeline with resumption support
- `evaluate.py` - Keyword-based refusal ratio evaluation
- `prepare_datasets.py` - Dataset preparation
- `outputs/` - All generation and evaluation results

## Key Decisions
- Using keyword-based refusal detection (paper uses WildGuard but keyword is simpler)
- Qwen3-8B with /no_think prefix to disable thinking mode
- Save every 10 samples for resilience
- Hyperparameters: α=4.5, λ=0.9, β=0.01, k=10, N=512, greedy decoding

## Remaining Work (priority order)
1. Finish orbench_hard AdaCD (~609 more samples, ~100 min)
2. Run evaluate.py on all 6 AdaCD outputs
3. Generate comparison table matching Table 2
4. Write reproduce.sh
5. Write REPORT.md
6. Final commit and end_task
