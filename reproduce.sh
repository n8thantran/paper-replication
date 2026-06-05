#!/bin/bash
# reproduce.sh - Reproduce AdaCD paper results (Table 2, Qwen3-8B row)
#
# This script:
# 1. Prepares all 6 datasets
# 2. Generates default (baseline) responses for all datasets
# 3. Generates AdaCD responses for all datasets
# 4. Evaluates all outputs and produces comparison table
#
# If outputs already exist, generation steps are skipped (resumable).
# Expected runtime: ~4-6 hours on a single GPU for full generation.
# For evaluation only (if outputs exist): ~1 minute.

set -e

echo "============================================"
echo "AdaCD Paper Replication - Reproduce Script"
echo "============================================"

# Install dependencies
pip install transformers torch datasets accelerate --quiet 2>/dev/null

# Step 1: Prepare datasets
echo ""
echo "Step 1: Preparing datasets..."
python3 prepare_datasets.py

# Check if all outputs already exist
ALL_EXIST=true
for ds in xstest_safe xstest_unsafe oktest jailbench advbench orbench_hard; do
    for method in default adacd; do
        if [ ! -f "outputs/${ds}_${method}.json" ]; then
            ALL_EXIST=false
            break 2
        fi
    done
done

if [ "$ALL_EXIST" = true ]; then
    echo ""
    echo "All output files already exist. Skipping generation steps."
    echo "Delete outputs/ to regenerate from scratch."
else
    # Step 2: Generate default responses
    echo ""
    echo "Step 2: Generating default (baseline) responses..."
    for ds in xstest_safe xstest_unsafe oktest jailbench; do
        if [ -f "outputs/${ds}_default.json" ]; then
            echo "  Skipping $ds default (already exists)"
        else
            echo "  Default generation: $ds"
            python3 generate.py --datasets $ds --method default --max_new_tokens 512
        fi
    done

    for ds in advbench orbench_hard; do
        SAMPLES=200
        if [ "$ds" = "orbench_hard" ]; then SAMPLES=300; fi
        if [ -f "outputs/${ds}_default.json" ]; then
            echo "  Skipping $ds default (already exists)"
        else
            echo "  Default generation: $ds ($SAMPLES samples)"
            python3 generate.py --datasets $ds --method default --max_new_tokens 512 --max_samples $SAMPLES
        fi
    done

    # Step 3: Generate AdaCD responses
    echo ""
    echo "Step 3: Generating AdaCD responses..."
    for ds in xstest_safe xstest_unsafe oktest jailbench; do
        if [ -f "outputs/${ds}_adacd.json" ]; then
            echo "  Skipping $ds adacd (already exists)"
        else
            echo "  AdaCD generation: $ds"
            python3 generate.py --datasets $ds --method adacd --max_new_tokens 512 \
                --alpha 4.5 --lambda_ 0.9 --beta 0.01 --k 10
        fi
    done

    for ds in advbench orbench_hard; do
        SAMPLES=200
        if [ "$ds" = "orbench_hard" ]; then SAMPLES=300; fi
        if [ -f "outputs/${ds}_adacd.json" ]; then
            echo "  Skipping $ds adacd (already exists)"
        else
            echo "  AdaCD generation: $ds ($SAMPLES samples)"
            python3 generate.py --datasets $ds --method adacd --max_new_tokens 512 \
                --alpha 4.5 --lambda_ 0.9 --beta 0.01 --k 10 --max_samples $SAMPLES
        fi
    done
fi

# Step 4: Evaluate all outputs
echo ""
echo "Step 4: Evaluating all outputs..."
python3 evaluate_all.py

# Step 5: Generate final results
echo ""
echo "Step 5: Generating final results..."
python3 generate_results.py

echo ""
echo "============================================"
echo "Done! Results saved to results/"
echo "  - results/summary.json"
echo "  - results/refusal_ratios.json"
echo "  - results/results_table.md"
echo "  - results/table2_comparison.txt"
echo "============================================"
