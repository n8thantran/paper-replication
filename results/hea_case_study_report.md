# HEA-Executor Analytical Report

## Task Description
Can the room-temperature plasticity of MoNbTaW refractory HEAs be optimized by adjusting compositional ratios? Identify the optimal six-element system from the 15-element palette.

## Database Overview
- **Element palette**: Al, Co, Cr, Cu, Fe, Mn, Mo, Nb, Ni, Ti, V, W, Zr, Ta, Hf (15 elements)
- **Total six-element combinations**: 5005
- **Descriptor dimensions per composition**: 194
- **Estimated full database size**: ~17.4 TB

## Query Results (Computation-Near-Data)
- **Matching combinations**: 55
- **Total samples analyzed**: 27500

## Combination Rankings (by % highly ductile)

| Rank | System | Samples | % Highly Ductile | Avg Score | Avg VEC | Pugh B/G |
|------|--------|---------|-----------------|-----------|---------|----------|
| 1 | Al-Hf-Mo-Nb-Ta-W | 500 | 10.20% | 0.1809 | 4.85 | 2.48 |
| 2 | Hf-Mo-Nb-Ta-W-Zr | 500 | 2.80% | 0.1257 | 5.00 | 2.49 |
| 3 | Al-Mo-Nb-Ta-W-Zr | 500 | 2.20% | 0.1281 | 4.83 | 2.44 |
| 4 | Cr-Hf-Mo-Nb-Ta-W | 500 | 1.20% | 0.1166 | 5.34 | 2.22 |
| 5 | Hf-Mo-Nb-Ta-Ti-W | 500 | 1.20% | 0.1188 | 4.99 | 2.46 |
| 6 | Al-Mo-Nb-Ni-Ta-W | 500 | 0.80% | 0.1106 | 5.86 | 2.39 |
| 7 | Hf-Mo-Nb-Ta-V-W | 500 | 0.80% | 0.1152 | 5.17 | 2.56 |
| 8 | Al-Cr-Mo-Nb-Ta-W | 500 | 0.60% | 0.1121 | 5.18 | 2.18 |
| 9 | Al-Fe-Mo-Nb-Ta-W | 500 | 0.60% | 0.1081 | 5.49 | 2.34 |
| 10 | Al-Mo-Nb-Ta-Ti-W | 500 | 0.60% | 0.1115 | 4.83 | 2.41 |
| 11 | Al-Mo-Nb-Ta-V-W | 500 | 0.60% | 0.1180 | 4.99 | 2.51 |
| 12 | Fe-Hf-Mo-Nb-Ta-W | 500 | 0.60% | 0.1161 | 5.68 | 2.40 |
| 13 | Fe-Mn-Mo-Nb-Ta-W | 500 | 0.60% | 0.1137 | 6.17 | 2.20 |
| 14 | Hf-Mo-Nb-Ni-Ta-W | 500 | 0.60% | 0.1151 | 5.98 | 2.46 |
| 15 | Al-Co-Mo-Nb-Ta-W | 500 | 0.40% | 0.1168 | 5.67 | 2.39 |
| 16 | Co-Hf-Mo-Nb-Ta-W | 500 | 0.40% | 0.1104 | 5.85 | 2.45 |
| 17 | Co-Mo-Nb-Ta-Ti-W | 500 | 0.40% | 0.1115 | 5.83 | 2.37 |
| 18 | Cu-Hf-Mo-Nb-Ta-W | 500 | 0.40% | 0.1091 | 6.15 | 2.51 |
| 19 | Mo-Nb-Ta-Ti-V-W | 500 | 0.40% | 0.1143 | 5.17 | 2.48 |
| 20 | Co-Cu-Mo-Nb-Ta-W | 500 | 0.20% | 0.1089 | 6.99 | 2.43 |

## Machine Learning Analysis

### Ductility Classification (RandomForest)
- **Samples**: 27500
- **5-Fold CV F1**: 0.1474 ± 0.0244
- **5-Fold CV Accuracy**: 0.9817 ± 0.0006
- **Positive rate (highly ductile)**: 0.50%

**Top Feature Importances:**
  - vec_avg: 0.1818
  - frac_Hf: 0.1469
  - cauchy_pressure: 0.0936
  - frac_Al: 0.0687
  - pugh_ratio: 0.0679

### Plasticity Regression (GradientBoosting)
- **5-Fold CV R²**: -0.0014 ± 0.0071
- **5-Fold CV MAE**: 2.4783 ± 0.0283

## Key Findings
- **Optimal system**: Al-Hf-Mo-Nb-Ta-W
- **Highly ductile fraction**: 10.20%
- **Baseline system (Ti-containing)**: Hf-Mo-Nb-Ta-Ti-W
- **Baseline highly ductile fraction**: 1.20%
- **Improvement factor**: 8.5x

## Data Sovereignty Compliance
- ✅ Remote-Execution-First: All computation at data side
- ✅ Minimal Data Return: Only statistics and rankings returned
- ✅ Database Isolation: No direct file access
- ✅ Restricted Descriptor Exposure: Only aggregate features returned
- ✅ Multi-User Shared Computing: Task-level isolation maintained