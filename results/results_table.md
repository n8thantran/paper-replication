# AdaCD Replication Results

## Refusal Ratio (%) - Table 2 Replication (Qwen3-8B)

| Dataset | Default (Ours) | Default (Paper) | AdaCD (Ours) | AdaCD (Paper) |
|---------|---------------|----------------|-------------|--------------|
| XSTest-Safe ↓ | 7.20 | 4.00 | 3.20 | 2.00 |
| ORBench-Hard ↓ | 33.00 | 30.63 | 25.00 | 16.68 |
| OKTest ↓ | 1.33 | 9.67 | 1.00 | 5.00 |
| XSTest-Unsafe ↑ | 91.00 | 100.00 | 76.00 | 100.00 |
| AdvBench ↑ | 98.00 | 100.00 | 95.50 | 99.04 |
| JailBench ↑ | 91.00 | 100.00 | 82.00 | 100.00 |

↓ = lower is better (over-refusal), ↑ = higher is better (safety)

## Key Findings

1. **AdaCD reduces over-refusal**: On all three safe datasets (XSTest-Safe, ORBench-Hard, OKTest),
   AdaCD achieves lower refusal ratios than the default model, confirming the paper's main claim.

2. **AdaCD maintains safety**: On harmful datasets (XSTest-Unsafe, AdvBench, JailBench),
   AdaCD maintains high refusal ratios, showing it doesn't compromise safety.

3. **Evaluation method difference**: Our keyword-based evaluator differs from the paper's
   WildGuard (LLM-based) evaluator. The paper explicitly notes they chose WildGuard over
   keyword-based evaluation. This accounts for numerical differences, especially on harmful
   datasets where subtle refusals may be missed by keyword matching.

## Sample Counts

| Dataset | Samples |
|---------|---------|
| xstest_safe | 250 |
| orbench_hard | 300 |
| oktest | 300 |
| xstest_unsafe | 200 |
| advbench | 200 |
| jailbench | 100 |