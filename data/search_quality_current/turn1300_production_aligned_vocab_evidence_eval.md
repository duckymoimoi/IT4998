# Production-Aligned Turn1300 Retrieval Evaluation

- CVs: 130
- ES index: `topcv_jobs_turn1300_rebuilt`
- Label source: `consensus_no_cohere` / `median`

| Variant | NDCG@10 | NDCG@20 | R@10 | R@20 | MRR | ESCO avg | ESCO CVs | Graph confirmed avg | Graph CVs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 planner hybrid | 0.3218 | 0.3201 | 0.1801 | 0.2814 | 0.5675 | 0.000 | 0 | 0.000 | 0 |
| P1 planner + controlled ESCO | 0.3258 | 0.3228 | 0.1812 | 0.2813 | 0.5837 | 0.077 | 10 | 0.000 | 0 |
| P2 planner + controlled ESCO + graph | 0.3248 | 0.3230 | 0.1806 | 0.2824 | 0.5760 | 0.077 | 10 | 0.077 | 5 |
