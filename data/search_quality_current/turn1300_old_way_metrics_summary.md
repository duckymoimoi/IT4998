# Turn 1300 Retrieval Metrics Summary

## Main old/canonical evaluation

- File: `data/search_quality_current/full_turn1300_final_canonical_tuned_kg_pool50.json`
- Pairs: `data/turn1300_v3_union_final/evaluation_pairs_2450_merged_4model.json`
- CVs: 130
- Ground-truth pairs: 2450
- Label source: `consensus_no_cohere`
- Pool size: 50

| Method | NDCG@10 | NDCG@20 | R@10 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| E1: BM25 Only | 0.5310 | 0.5172 | 0.3088 | 0.4642 | 0.7729 |
| E2: Hybrid BM25+kNN | 0.6535 | 0.6352 | 0.3854 | 0.5853 | 0.8249 |
| E2t: Hybrid BM25+kNN tuned | **0.6843** | **0.6610** | **0.4256** | **0.6243** | 0.8080 |
| E3: Hybrid + ESCO | 0.6583 | 0.6365 | 0.3907 | 0.5886 | **0.8291** |
| E3t: Hybrid + ESCO tuned | 0.6740 | 0.6570 | 0.4138 | 0.6189 | 0.8103 |
| E3g-basic: Hybrid + ESCO + KG tuned | 0.6709 | 0.6563 | 0.4113 | 0.6182 | 0.8116 |
| E3g-full: Hybrid + ESCO + KG tuned | 0.6705 | 0.6561 | 0.4117 | 0.6184 | 0.8157 |

## Normalized-field vocabulary run

- File: `data/search_quality_current/turn1300_final_no_cohere_normalized_fields.json`
- Pairs: `data/turn1300_v3_union_final/evaluation_pairs_2450_merged_4model.json`
- CVs: 130
- Ground-truth pairs: 2450
- Label source: `consensus_no_cohere`
- Pool size: 50

| Method | NDCG@10 | NDCG@20 | R@10 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| E1: BM25 Only | 0.4517 | 0.4523 | 0.2594 | 0.4120 | 0.7073 |
| E2: Hybrid BM25+kNN | 0.4723 | 0.4849 | 0.2723 | 0.4488 | **0.7326** |
| E2t: Hybrid BM25+kNN tuned | 0.4689 | 0.4895 | 0.2726 | **0.4606** | 0.7027 |
| E3: Hybrid + ESCO | 0.4814 | 0.4852 | 0.2818 | 0.4526 | 0.7045 |
| E3t: Hybrid + ESCO tuned | **0.4873** | **0.4950** | **0.2828** | 0.4580 | 0.7242 |

## Earlier remapped 1300-job evaluation

- File: `data/search_quality_current/ablation_turn1300_after_expired_fix_no_partial.json`
- Pairs: `data/evaluation_pairs_1300_v2_remapped_to_turn1300.json`
- CVs: 104
- Ground-truth pairs: 606
- Pool size: 50

| Method | NDCG@10 | NDCG@20 | R@10 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| E1: BM25 Only | 0.6623 | 0.7013 | 0.6732 | 0.7644 | 0.8301 |
| E2: Hybrid BM25+kNN | 0.6739 | 0.7105 | 0.7008 | 0.7831 | 0.7976 |
| E2t: Hybrid BM25+kNN tuned | 0.7600 | 0.8119 | 0.8051 | 0.9247 | 0.8461 |
| E3: Hybrid + ESCO | 0.6926 | 0.7202 | 0.6998 | 0.7605 | 0.8499 |
| E3t: Hybrid + ESCO tuned | **0.8106** | **0.8436** | **0.8692** | 0.9429 | 0.8926 |
| E3g-basic: Hybrid + ESCO + KG tuned | 0.8083 | 0.8430 | 0.8671 | **0.9443** | **0.8962** |

## Note

For the thesis, the safest main table is the 2450-pair canonical/no-Cohere evaluation. The normalized-field vocabulary run is better framed as a residual/pooling-bias analysis because it changes the representation and retrieves many top-ranked unjudged pairs outside the original pool.
