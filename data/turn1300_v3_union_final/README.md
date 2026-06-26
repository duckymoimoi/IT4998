# Turn 1300 V3 Union Final Dataset

Thu muc nay gom bo dataset danh gia cuoi cho thuc nghiem Turn 1300 union pooling.

## File chinh

- `evaluation_pairs_2450_merged_4model.json`: 2.450 cap CV-job da ghep day du annotation cua 4 model (`claude`, `cohere`, `gptoss`, `llama`) va consensus.
- `labels_2450_compact.json`: ban label gon cho 2.450 cap, co ca nhan 4-model va nhan no-Cohere.
- `labels_2450_compact.csv`: ban CSV cua label gon.
- `agreement_summary.json`: thong ke bat dong giua model, gom ca 4-model va bo 3 model khong Cohere.

## File nguon kem theo

- `topcv_balanced_1300_rebuilt_20260602.csv`: tap 1.300 job.
- `evaluation_cvs_130_canonical.json`: tap 130 CV dung de tao va cham 2.450 pair.
- `evaluation_pairs_1300_v3_union_pooling_trial_top10_E1_E2_E3_E2t_E3t.json`: pair pool goc 2.450 cap.
- `annotations_*_merged_2450.json`: annotation da merge rieng cho tung model, moi file co du 2.450 cap.

## Nguon annotation sau khi clean

Sau khi don thu muc, cac annotation tho theo tung dot da duoc thay bang cac file `annotations_*_merged_2450.json` trong thu muc nay. Cac file merge nay la nguon nen dung neu can kiem tra lai nhan cua tung model.

## Nhan khuyen nghi

Nhan chinh nen dung:

```text
consensus_4model.relevance_score.median
```

Neu can kiem tra do nhay khi bo Cohere, dung:

```text
consensus_no_cohere.relevance_score.median
```

Nguong relevant mac dinh: `>= 5`.
