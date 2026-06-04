#!/bin/bash
# ============================================================================
# OpenAaaS Replication - reproduce.sh
# Reproduces all key results from the paper:
#   "OpenAaaS: An Open-Source Agent-as-a-Service Framework for 
#    Agentic Materials Informatics"
# ============================================================================

set -e

echo "============================================================================"
echo "OpenAaaS Replication - Reproducing All Key Results"
echo "============================================================================"
echo ""

# Create results directory
mkdir -p /workspace/results

# Install dependencies
echo "[1/6] Installing dependencies..."
pip install flask duckdb pandas numpy scikit-learn pyarrow --quiet 2>/dev/null
echo "  ✓ Dependencies installed"
echo ""

# ============================================================================
# Result 1: HEA Case Study (Paper Section 4, Table 1)
# ============================================================================
echo "[2/6] Running HEA Case Study..."
echo "  This reproduces the main case study from Section 4:"
echo "  - 5,005 six-element combinations from 15-element palette"
echo "  - 55 MoNbTaW-containing combinations screened"
echo "  - Optimal system identification with ductility improvement"
echo ""

cd /workspace
python3 -c "
import sys, json, time
sys.path.insert(0, '/workspace')
from executors.hea_executor import HEAExecutor

print('Initializing HEA Executor...')
executor = HEAExecutor()

task = {
    'description': 'Screen MoNbTaW-containing six-element HEAs for ductility',
    'required_elements': ['Mo', 'Nb', 'Ta', 'W']
}

print('Running full HEA screening pipeline...')
result = executor.execute(task)

# Save results
with open('/workspace/results/hea_case_study_results.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)

# Generate report
opt = result['optimal_system']
baseline = result['baseline_system']
print()
print('=' * 60)
print('HEA Case Study Results')
print('=' * 60)
print(f'  Total 6-element combinations: 5,005')
print(f'  MoNbTaW-containing screened: {result[\"n_combinations_screened\"]}')
print(f'  Total samples analyzed: {result[\"total_samples_analyzed\"]}')
print(f'  Optimal system: {opt[\"combination_id\"]}')
print(f'  Optimal ductile fraction: {opt[\"pct_highly_ductile\"]}%')
print(f'  Baseline (MoNbTaW+Ti+Hf): {baseline[\"combination_id\"]}')
print(f'  Baseline ductile fraction: {baseline[\"pct_highly_ductile\"]}%')
print(f'  Improvement factor: {result[\"improvement_factor\"]}x')
print(f'  Result size: {result[\"result_size_mb\"]:.3f} MB (vs 17.4 TB raw)')
print()
print('Paper comparison:')
print(f'  Paper optimal: Al-Hf-Mo-Nb-Ta-W (10.20% ductile, 10.7x improvement)')
print(f'  Our result:    {opt[\"combination_id\"]} ({opt[\"pct_highly_ductile\"]}% ductile, {result[\"improvement_factor\"]}x improvement)')
print('=' * 60)
"
echo "  ✓ HEA case study results saved to results/hea_case_study_results.json"
echo ""

# ============================================================================
# Result 2: AlphaAgent Evaluation (Paper Table 2)
# ============================================================================
echo "[3/6] Running AlphaAgent Evaluation (Table 2)..."
echo "  This reproduces the evaluation scores from Table 2:"
echo "  - Deep analytical Q&A evaluation"
echo "  - General Q&A evaluation"
echo ""

python3 -c "
import sys
sys.path.insert(0, '/workspace')
from executors.alpha_agent_executor import run_alpha_agent_evaluation
run_alpha_agent_evaluation()
"
echo "  ✓ AlphaAgent evaluation saved to results/alpha_agent_evaluation.json"
echo "  ✓ Table 2 saved to results/table2.txt"
echo ""

# ============================================================================
# Result 3: Architectural Comparison (Paper Table 3)
# ============================================================================
echo "[4/6] Generating Architectural Comparison (Table 3)..."

python3 /workspace/generate_table3.py
echo "  ✓ Table 3 saved to results/table3.txt, table3.md, table3.json"
echo ""

# ============================================================================
# Result 4: Server Routing Overhead
# ============================================================================
echo "[5/6] Measuring Server Routing Overhead..."

python3 /workspace/measure_routing_overhead.py
echo "  ✓ Routing overhead saved to results/routing_overhead.json"
echo ""

# ============================================================================
# Result 5: Integration Test (Full 3-Tier Architecture)
# ============================================================================
echo "[6/6] Running Full Integration Test..."

python3 /workspace/integration_test.py
echo ""

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "All Results Generated Successfully!"
echo "============================================================================"
echo ""
echo "Results directory contents:"
ls -la /workspace/results/
echo ""
echo "Key files:"
echo "  results/hea_case_study_results.json  - HEA case study (Section 4)"
echo "  results/alpha_agent_evaluation.json  - AlphaAgent scores (Table 2)"
echo "  results/table2.txt                   - Table 2 formatted"
echo "  results/table3.txt                   - Table 3 formatted"
echo "  results/table3.md                    - Table 3 markdown"
echo "  results/routing_overhead.json        - Server routing overhead"
echo "  results/hea_case_study_report.md     - HEA case study report"
echo ""
echo "============================================================================"
echo "Replication Complete"
echo "============================================================================"
