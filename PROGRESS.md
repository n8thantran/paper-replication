# OpenAaaS Replication Progress

## Current Phase
Reading paper and creating implementation plan.

## Paper Summary
OpenAaaS is a hierarchical, distributed Agent-as-a-Service framework for materials informatics with 3 tiers:
1. Master Agent Layer (LLM agents for task decomposition)
2. Network Hub (HTTP server with SQLite for routing)
3. Network Node (Agent Core for near-data execution)

## Key Results to Reproduce
1. **Table 2**: AlphaAgent evaluation scores (4.66/5.0 deep analytical, 4.46/5.0 general)
2. **Table 3**: Architectural comparison table
3. **HEA Case Study**: 
   - 5,005 six-element combinations from 15-element palette
   - 55 MoNbTaW-containing combinations screened
   - Al-Mo-Nb-Ta-W-Hf identified as optimal (9.80% highly ductile, 10.7x improvement)
   - Server routing overhead ~550ms
   - Returned data ~2.3MB vs 17.4TB raw

## Implementation Plan
- [ ] 1. OpenAaaS Server (Python/Flask with SQLite)
  - Service registration, task routing, heartbeat, file relay, auth
- [ ] 2. Agent Core (task polling, execution, result reporting)
- [ ] 3. AlphaAgent Executor (RAG pipeline with evidence grounding)
  - Intent rewriting, retrieval, evidence validation, reporting
- [ ] 4. HEA Executor (descriptor database, DuckDB querying, ML)
  - 15-element palette, permutation-invariant matching, scikit-learn
- [ ] 5. Evaluation & Results
  - Reproduce Table 2, Table 3, HEA case study results
- [ ] 6. reproduce.sh and REPORT.md

## Key Decisions
- Server in Python (paper uses Rust, but Python is more practical for ML integration)
- HEA database: scaled-down version demonstrating pipeline (17.4TB infeasible)
- AlphaAgent: demonstrate pipeline without 300K paper index

## Completed Work
(none yet)

## Failed Approaches
(none yet)
