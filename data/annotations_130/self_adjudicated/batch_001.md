# Self Adjudication Batch 001

- Method: Codex self-adjudication, no external API, no regex/rule-based scoring.
- Source: `data/evaluation_pairs_1300_v2.json` + `data/annotations_130/annotations_*.json`
- Cases: 1-20 of 431 high-disagreement pairs.
- Threshold: any dimension with max-min > 4.

| # | Pair key | High dims | Final rel | Final skills | Final exp | Final edu | Note |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `bds_L1_0feeda23a58657fb3da9dd93842f8249_3` | skills, exp | 6 | 7 | 6 | 8 | Core Sales BDS/B2C match, but lacks telesales/online sales and 6 months experience. |
| 2 | `bds_L1_649e49a7851237ec04913b846e2760dd_15` | skills | 8 | 8 | 10 | 8 | Sales BDS no-experience job; CV matches core skills. |
| 3 | `bds_L1_6f45f047db1ac4041f38e7f48d5764e4_13` | exp | 5 | 5 | 4 | 8 | Related BDS background, but lacks Sales Admin/Excel/Sheet and 1 year admin experience. |
| 4 | `bds_L1_e43ade755d9156bda5212d5e23303442_11` | skills | 7 | 7 | 10 | 8 | Good BDS/B2C fit, missing some B2B/Direct Sales depth. |
| 5 | `bds_L2_24470b97617841c1f85862f4c04ce539_13` | exp | 5 | 4 | 4 | 8 | Sales BDS experience exists but lacks team management for head role. |
| 6 | `bds_L2b_6bfd53767bb4463720b123ebc03a27c2_4` | exp | 7 | 7 | 6 | 8 | Related BDS/project skills, mildly overqualified for under-1-year junior job. |
| 7 | `bds_L2b_70615d0a6923e30854ea35dba94c7202_8` | skills | 4 | 4 | 3 | 8 | Investment development role needs 5 years and construction/legal/software skills; CV is mainly sales. |
| 8 | `bds_L3_6bfd53767bb4463720b123ebc03a27c2_5` | exp | 6 | 7 | 5 | 8 | Strong BDS/project fit, but 3 years is overqualified for under-1-year admin role. |
| 9 | `bds_L3b_4ccda888dc185e53c6595de192044610_5` | exp | 6 | 5 | 5 | 8 | Related BDS management, not direct procedures/contracts; slightly over 1-3 year target. |
| 10 | `bds_L3b_6bfd53767bb4463720b123ebc03a27c2_3` | skills, exp | 6 | 7 | 4 | 8 | Strong BDS/project skills, but 4 years manager profile is too senior for junior admin role. |
| 11 | `bds_L4_6bfd53767bb4463720b123ebc03a27c2_5` | exp | 6 | 7 | 4 | 8 | Relevant skills, but 6 years is heavy overqualification for under-1-year role. |
| 12 | `bds_L4_7d079b4b1f713fe7edd7cfb413cd3aef_4` | exp | 6 | 8 | 4 | 8 | Strong BDS sales match; no-experience job requires overqualification penalty. |
| 13 | `bds_L4_921b61b86e73bd5c2a0fa22475f0d82e_10` | exp | 6 | 7 | 4 | 8 | Actual JD is BDS sales; extracted tech skills are noisy, but seniority is too high. |
| 14 | `bds_L4_f3285daaa63b2a31d61f0cbb4d9e3dd9_15` | exp | 7 | 8 | 4 | 8 | Core sales skills strong; 6 years is overqualified for no-experience job. |
| 15 | `bds_L4b_6bfd53767bb4463720b123ebc03a27c2_6` | rel, skills, exp | 5 | 7 | 3 | 8 | Very senior director profile for junior admin role; skills related but level mismatch is large. |
| 16 | `bds_L4b_70615d0a6923e30854ea35dba94c7202_10` | skills | 6 | 5 | 7 | 9 | Senior BDS experience and education fit, but lacks investment/construction/legal/software specifics. |
| 17 | `bds_L4b_7d079b4b1f713fe7edd7cfb413cd3aef_5` | exp | 6 | 7 | 4 | 8 | Sales BDS match, but 7 years management is overqualified for no-experience staff role. |
| 18 | `bds_L4b_921b61b86e73bd5c2a0fa22475f0d82e_7` | exp | 5 | 6 | 4 | 8 | JD is BDS sales but skill fields are noisy; seniority remains too high. |
| 19 | `bds_P2_d3baab157fe54f4c6135b0ef11d38b42_10` | edu | 5 | 5 | 9 | 4 | Customer-service experience transfers to reception, but Trung cấp misses Cao đẳng/ĐH requirement. |
| 20 | `bds_P2_e64a793e79b507d7cfba0808d84d5117_3` | exp | 4 | 3 | 5 | 3 | Some inventory exposure, but not warehouse admin/logistics; lacks WMS/SAP/Excel and education requirement. |

