# Turn 1300 final retrieval evaluation - normalized job fields

- Dataset: `data/turn1300_v3_union_final/evaluation_pairs_2450_merged_4model.json`
- CVs: 130; GT pairs: 2450; pool size: 50
- Label source: `consensus_no_cohere` / `median`; Cohere excluded
- Output JSON: `data/search_quality_current/turn1300_final_no_cohere_normalized_fields_with_unjudged_fullcap.json`

| Experiment | NDCG@10 | NDCG@20 | R@10 | R@20 | MRR | ?NDCG@10 vs E1 |
|---|---:|---:|---:|---:|---:|---:|
| E1: BM25 Only | 0.4517 | 0.4523 | 0.2594 | 0.4120 | 0.7073 | +0.0000 |
| E2: Hybrid BM25+kNN | 0.4723 | 0.4849 | 0.2723 | 0.4488 | 0.7326 | +0.0206 |
| E2t: Hybrid BM25+kNN tuned | 0.4689 | 0.4895 | 0.2726 | 0.4606 | 0.7027 | +0.0172 |
| E3: Hybrid + ESCO | 0.4814 | 0.4852 | 0.2818 | 0.4526 | 0.7045 | +0.0297 |
| E3t: Hybrid + ESCO tuned | 0.4873 | 0.4950 | 0.2828 | 0.4580 | 0.7242 | +0.0356 |

Best by NDCG@10: **E3t: Hybrid + ESCO tuned** (0.4873).

Interpretation: normalized job fields improve hybrid retrieval over BM25; ESCO + tuned RRF is best on NDCG@10 in this run, while default hybrid has the best MRR among non-ESCO methods.
