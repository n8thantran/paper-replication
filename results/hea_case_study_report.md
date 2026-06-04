# HEA-Executor Analytical Report

## Task Description
Can the room-temperature plasticity of MoNbTaW refractory HEAs be optimized by adjusting compositional ratios? Identify the optimal six-element system from the 15-element palette.

## Database Overview
- **Element palette**: Al, Co, Cr, Cu, Fe, Mn, Mo, Nb, Ni, Ti, V, W, Zr, Ta, Hf (15 elements)
- **Total six-element combinations**: 5005
- **Descriptor dimensions**: 194

## Query Results
- **Matching combinations**: 55
- **Total samples analyzed**: 11000

## Combination Rankings (by % highly ductile)

| Rank | Combination | Samples | % Highly Ductile | Avg Ductility Score | Avg VEC | Pugh Ratio |
|------|-------------|---------|-----------------|--------------------|---------|-----------| 
| 1 | Al-Mo-Nb-Ta-V-W | 200 | 100.00% | 0.7510 | 5.02 | 2.50 |
| 2 | Al-Mo-Nb-Ta-Ti-W | 200 | 99.50% | 0.7495 | 4.85 | 2.40 |
| 3 | Mo-Nb-Ta-Ti-V-W | 200 | 99.50% | 0.7443 | 5.18 | 2.47 |
| 4 | Co-Cu-Mo-Nb-Ta-W | 200 | 98.00% | 0.7497 | 7.05 | 2.41 |
| 5 | Cu-Mo-Nb-Ni-Ta-W | 200 | 97.50% | 0.7509 | 7.14 | 2.42 |
| 6 | Hf-Mo-Nb-Ta-Ti-W | 200 | 97.50% | 0.6957 | 4.99 | 2.46 |
| 7 | Co-Mo-Nb-Ta-V-W | 200 | 97.00% | 0.6795 | 5.99 | 2.45 |
| 8 | Cr-Mn-Mo-Nb-Ta-W | 200 | 97.00% | 0.6881 | 5.84 | 2.04 |
| 9 | Fe-Mn-Mo-Nb-Ta-W | 200 | 97.00% | 0.6835 | 6.16 | 2.18 |
| 10 | Al-Cr-Mo-Nb-Ta-W | 200 | 96.50% | 0.7009 | 5.18 | 2.19 |
| 11 | Al-Hf-Mo-Nb-Ta-W | 200 | 96.50% | 0.6948 | 4.82 | 2.49 |
| 12 | Cu-Mo-Nb-Ta-Ti-W | 200 | 96.50% | 0.6954 | 6.16 | 2.44 |
| 13 | Cr-Cu-Mo-Nb-Ta-W | 200 | 96.00% | 0.6935 | 6.50 | 2.20 |
| 14 | Cr-Mo-Nb-Ta-V-W | 200 | 96.00% | 0.6867 | 5.51 | 2.24 |
| 15 | Fe-Mo-Nb-Ta-Ti-W | 200 | 96.00% | 0.6769 | 5.66 | 2.32 |
| 16 | Al-Co-Mo-Nb-Ta-W | 200 | 95.50% | 0.6879 | 5.62 | 2.39 |
| 17 | Al-Mn-Mo-Nb-Ta-W | 200 | 95.50% | 0.6924 | 5.32 | 2.26 |
| 18 | Cu-Fe-Mo-Nb-Ta-W | 200 | 95.50% | 0.7198 | 6.81 | 2.36 |
| 19 | Cu-Mn-Mo-Nb-Ta-W | 200 | 95.50% | 0.7040 | 6.68 | 2.28 |
| 20 | Hf-Mo-Nb-Ta-W-Zr | 200 | 95.50% | 0.6798 | 5.00 | 2.48 |

## Machine Learning Analysis

### Ductility Classification
- **Model**: RandomForestClassifier
- **Samples**: 11000
- **5-Fold CV F1**: 0.9265 ± 0.0017
- **5-Fold CV Accuracy**: 0.8695 ± 0.0025
- **Positive rate (highly ductile)**: 90.96%

**Top Feature Importances:**
  - delta_r: 0.3394
  - vec_avg: 0.1156
  - frac_Zr: 0.0527
  - S_mix: 0.0478
  - Tm_avg: 0.0395

### Plasticity Regression
- **Model**: GradientBoostingRegressor
- **5-Fold CV R²**: 0.2144 ± 0.0283
- **5-Fold CV MAE**: 2.3239 ± 0.0252
- **Mean plasticity**: 27.33%

## Key Findings
- **Optimal system**: Al-Mo-Nb-Ta-V-W
- **Highly ductile fraction**: 100.00%
- **Baseline system**: Al-Mo-Nb-Ta-Ti-W (99.50%)
- **Improvement factor**: 1.0x