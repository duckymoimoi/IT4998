# Production-Aligned Turn1300 Retrieval Evaluation

- CVs: 130
- ES index: `topcv_jobs_turn1300_rebuilt`
- Label source: `consensus_no_cohere` / `median`

| Variant | NDCG@10 | NDCG@20 | R@10 | R@20 | MRR | ESCO avg | ESCO CVs | Graph confirmed avg | Graph CVs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 planner hybrid | 0.3068 | 0.3107 | 0.1774 | 0.2808 | 0.5327 | 0.000 | 0 | 0.000 | 0 |
| P1 planner + controlled ESCO | 0.3080 | 0.3124 | 0.1772 | 0.2816 | 0.5390 | 0.077 | 10 | 0.000 | 0 |
| P2 planner + controlled ESCO + graph | 0.3079 | 0.3126 | 0.1772 | 0.2819 | 0.5390 | 0.077 | 10 | 0.038 | 5 |
