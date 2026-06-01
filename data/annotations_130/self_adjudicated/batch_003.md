# Self Adjudication Batch 003

- Method: Codex self-adjudication, no external API, no regex/rule-based scoring.
- Source: `data/evaluation_pairs_1300_v2.json` + `data/annotations_130/annotations_*.json`
- Cases: 41-60 of 431 high-disagreement pairs.
- Threshold: any dimension with max-min > 4.

## Batch Summary

Batch 003 contains many cross-domain cases: logistics/warehouse profiles matched to teaching/design jobs, developer profiles matched to CSKH/banking/F&B operations, and education profiles matched to Pilates or AI/ML roles.

Main disagreement pattern:

- `skills_match`: 8/20 cases.
- `experience_match`: 8/20 cases.
- `education_match`: 6/20 cases.

Adjudication principles used:

- Job title/domain overrides noisy extracted skill fields. Example: `Giáo Viên Tiếng Anh` with `Excel/AutoCAD/Python/SAP` extracted is still primarily an English-teacher job.
- For cross-domain roles, bậc học alone is not enough when the JD explicitly requires a specialized field. Example: AI/ML Lead requires CS/AI/Data degree; English degree gets capped.
- For IT roles, a single overlapping programming language is partial evidence, not full skills match. Missing frameworks, databases, cloud/devops, and production stack should keep skills around 3-5.

| # | Pair key | High dims | Final rel | Final skills | Final exp | Final edu | Note |
|---:|---|---|---:|---:|---:|---:|---|
| 41 | `cskh_P2_31f94b71b9ece5490c0d975496a785f7_10` | skills, exp | 1 | 1 | 2 | 1 | English-teacher title; warehouse/accounting CV has no teaching or English evidence. |
| 42 | `cskh_P2_bc10e325808bd3a8936168eea7938483_6` | exp, edu | 2 | 2 | 2 | 4 | AutoCAD/SAP are minor overlaps; no design/architecture experience or relevant education. |
| 43 | `cskh_P2_e92fe0f25fadb955d7462df748dbfb0b_4` | edu | 1 | 0 | 0 | 1 | Graphic Designer lead requires design degree/tools and >5 years; CV misses all core requirements. |
| 44 | `cskh_P_25c96711e7ff20a97e4199f4474cfb90_10` | skills | 5 | 4 | 7 | 8 | Python/programming helps, but AI/API/data-flow/tools are missing. |
| 45 | `cskh_P_393569e29b4b58bc0f81851017ed3cd8_11` | exp | 3 | 2 | 5 | 5 | No-exp banking sales still needs finance/sales/CRM/office fit; developer CV is weak. |
| 46 | `cskh_P_3d70186b1b7186e76993b29dca7ae421_8` | skills | 6 | 5 | 10 | 10 | Python and 2-year software background match minimum, but most web stack is absent. |
| 47 | `cskh_P_3ef4c70e843c8b027eb6f49cf3f4257e_7` | edu | 2 | 1 | 1 | 7 | Degree level passes, but IT/mobile background fails F&B operations requirements. |
| 48 | `cskh_P_6ab62df42ffa0205400bb950ba3b7ab6_6` | skills | 5 | 4 | 5 | 10 | Has Java and CS degree, but lacks 3-year Java web stack and cloud/database requirements. |
| 49 | `cskh_P_90c81ec8e7e2d0fde869ddf09b995492_13` | skills | 3 | 2 | 5 | 7 | CSKH English role; developer CV lacks English/Excel/CSKH evidence. |
| 50 | `cskh_P_fadbf9fddd686b0aba832143a92a3355_15` | skills | 3 | 2 | 3 | 8 | Hospital CSKH needs communication/problem solving/1 year CSKH; developer CV is off-domain. |
| 51 | `gd_L1_46b90624d63dfb96beb964fc7353948f_13` | exp | 3 | 2 | 2 | 7 | Pilates role requires anatomy/Pilates and 1 year; education fresher lacks domain skills. |
| 52 | `gd_L1_6c8d2cb86d40fdbf26901c625dcd08ca_4` | exp | 4 | 3 | 0 | 10 | Education degree fits, but no 3-year relevant science teaching experience. |
| 53 | `gd_L1_d8513f4d81b4553be6d704e58e7e0bbc_8` | exp | 5 | 5 | 2 | 10 | Related education background/tools, but no 2-year curriculum development experience. |
| 54 | `gd_L1_dce521390d3e06af94978332ba4636a1_5` | skills | 5 | 4 | 10 | 7 | Fresher is fine for under-1-year job, but lacks sales/B2B/direct-sales/IT-training fit. |
| 55 | `gd_L1_eeca4abe7e03fd2ec6a45bf6f3671cc2_15` | skills | 5 | 2 | 10 | 7 | Title says English teacher; CV has education but no English-teaching evidence in skills/degree. |
| 56 | `gd_L1b_46b90624d63dfb96beb964fc7353948f_13` | exp, edu | 4 | 3 | 5 | 7 | Teaching support transfers partly to instruction, but no Pilates/anatomy/STOTT. |
| 57 | `gd_L1b_eeca4abe7e03fd2ec6a45bf6f3671cc2_15` | skills | 7 | 6 | 9 | 10 | English degree and teaching support fit title, but empty JD prevents full skills confidence. |
| 58 | `gd_L2_46b90624d63dfb96beb964fc7353948f_12` | edu | 4 | 3 | 5 | 7 | Teaching experience helps communication only; Pilates/anatomy domain missing. |
| 59 | `gd_L2b_f224329725039dacca80697c36a5b0fe_12` | edu | 1 | 1 | 1 | 5 | AI/ML Lead requires CS/AI/Data degree and 5+ years; English teaching CV is off-domain. |
| 60 | `gd_L3_4cd7d72832462d7bd257a9039a0014f3_7` | exp | 6 | 7 | 5 | 10 | Education sales and IELTS fit, but 3-year training manager is overqualified for part-time junior online sales. |

## Experimental Notes

This batch shows why raw annotation agreement is low on noisy extracted fields:

- Several jobs have title/domain contradicting extracted technical skills.
- Some models over-trust structured fields like `job_technical_skills`; manual adjudication has to use JD title and requirements.
- Education scoring needs two modes: bậc học for generic requirements, and major/domain matching for specialist roles.

