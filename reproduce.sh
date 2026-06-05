#!/bin/bash
# reproduce.sh - Reproduce AdaCD paper results (Table 2, Qwen3-8B row)
#
# This script:
# 1. Prepares all 6 datasets
# 2. Generates default (baseline) responses for all datasets
# 3. Generates AdaCD responses for all datasets
# 4. Evaluates all outputs and produces comparison table
#
# Expected runtime: ~3-4 hours on a single GPU (Qwen3-8B)
# For faster testing, reduce --max_samples in the generation steps.

set -e

echo "============================================"
echo "AdaCD Paper Replication - Reproduce Script"
echo "============================================"

# Install dependencies
pip install transformers torch datasets accelerate --quiet

# Step 1: Prepare datasets
echo ""
echo "Step 1: Preparing datasets..."
python3 prepare_datasets.py

# Step 2: Generate default responses
echo ""
echo "Step 2: Generating default (baseline) responses..."
for ds in xstest_safe xstest_unsafe oktest jailbench; do
    echo "  Default generation: $ds"
    python3 generate.py --datasets $ds --method default --max_new_tokens 512
done

# Larger datasets with sample limits
echo "  Default generation: advbench (200 samples)"
python3 generate.py --datasets advbench --method default --max_new_tokens 512 --max_samples 200
echo "  Default generation: orbench_hard (300 samples)"
python3 generate.py --datasets orbench_hard --method default --max_new_tokens 512 --max_samples 300

# Step 3: Generate AdaCD responses
echo ""
echo "Step 3: Generating AdaCD responses..."
for ds in xstest_safe xstest_unsafe oktest jailbench; do
    echo "  AdaCD generation: $ds"
    python3 generate.py --datasets $ds --method adacd --max_new_tokens 512 \
        --alpha 4.5 --lambda_ 0.9 --beta 0.01 --k 10
done

echo "  AdaCD generation: advbench (200 samples)"
python3 generate.py --datasets advbench --method adacd --max_new_tokens 512 \
    --alpha 4.5 --lambda_ 0.9 --beta 0.01 --k 10 --max_samples 200
echo "  AdaCD generation: orbench_hard (300 samples)"
python3 generate.py --datasets orbench_hard --method adacd --max_new_tokens 512 \
    --alpha 4.5 --lambda_ 0.9 --beta 0.01 --k 10 --max_samples 300

# Step 4: Evaluate all outputs
echo ""
echo "Step 4: Evaluating all outputs..."
python3 evaluate_all.py

echo ""
echo "============================================"
echo "Done! Results saved to results/"
echo "============================================"
