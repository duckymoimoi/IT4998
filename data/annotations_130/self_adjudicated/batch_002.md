# Self Adjudication Batch 002

- Method: Codex self-adjudication, no external API, no regex/rule-based scoring.
- Source: `data/evaluation_pairs_1300_v2.json` + `data/annotations_130/annotations_*.json`
- Cases: 21-40 of 431 high-disagreement pairs.
- Threshold: any dimension with max-min > 4.

## Batch Summary

Batch 002 contains the transition from `Bất động sản` partial/cross-domain cases into `Chăm sóc khách hàng` senior-vs-junior cases.

Main disagreement pattern:

- `experience_match` dominates: 16/20 cases.
- `education_match`: 4/20 cases, mostly due to whether wrong-major degrees should still pass bậc học.
- `skills_match`: 2/20 cases.

Key adjudication decisions:

- For senior CSKH profiles applying to no-experience or under-1-year CSKH jobs, skills can remain moderate-high, but `experience_match` is capped around 4-5 for overqualification.
- For cross-domain IT profiles applying to bank sales, operations, clinic management, or HR roles, education is not automatically 10 just because the candidate has a university degree. If the JD names a domain or the role is clearly domain-specific, wrong major is penalized.
- Empty JD fields are handled conservatively: when no education requirement is visible, education is not heavily penalized, but relevance/skills/experience stay low if the title/domain is clearly mismatched.

| # | Pair key | High dims | Final rel | Final skills | Final exp | Final edu | Note |
|---:|---|---|---:|---:|---:|---:|---|
| 21 | `bds_P2_f6238a52435180210acdd42e0ec67300_14` | edu | 4 | 3 | 3 | 3 | Service/inventory background is weak for 4-year logistics sales; Trung cấp misses Cao đẳng. |
| 22 | `bds_P_393569e29b4b58bc0f81851017ed3cd8_5` | exp | 3 | 2 | 5 | 5 | No-experience finance sales job, but IT background is weak and wrong field. |
| 23 | `bds_P_3d70186b1b7186e76993b29dca7ae421_2` | skills | 6 | 5 | 10 | 10 | IT profile matches Python/JS and 2 years, but lacks much of the stack. |
| 24 | `bds_P_3ef4c70e843c8b027eb6f49cf3f4257e_1` | edu | 2 | 1 | 1 | 7 | Degree level passes generic requirement, but IT profile fails F&B operations. |
| 25 | `bds_P_b56466a49185c5e7ac3a46dce4c4f9c2_9` | exp, edu | 2 | 0 | 2 | 7 | JD is mostly empty beyond HR title; do not infer strong match. |
| 26 | `bds_P_d040b13ab80f50ff495b8afd443c0f52_12` | skills, exp | 4 | 3 | 5 | 10 | Has JavaScript/software background, lacks frontend framework depth and 3+ years. |
| 27 | `bds_P_d12916f218d2f899ccf9ed77e6566cec_15` | edu | 2 | 1 | 1 | 5 | Clinic manager role needs healthcare/management; IT degree is wrong domain. |
| 28 | `cskh_L1_21e374ac9c4191dfb45bdc896cc76ac5_12` | exp | 5 | 4 | 3 | 7 | Some CSKH basics, but no 1-year telesales/sales consulting experience. |
| 29 | `cskh_L1_fadbf9fddd686b0aba832143a92a3355_9` | exp | 6 | 6 | 4 | 8 | Good CSKH soft skills, but lacks required 1 year professional experience. |
| 30 | `cskh_L3_124f4256f0d6a862038089b366463a3e_1` | exp | 6 | 6 | 5 | 8 | CSKH and English fit, but lacks software knowledge and is overqualified for under-1-year role. |
| 31 | `cskh_L3_12dc6cb1ab0030d1747743ac523bbcce_6` | exp | 5 | 5 | 5 | 8 | Partial CSKH/Excel fit, missing CRM/Salesforce; 3 years is above junior level. |
| 32 | `cskh_L3_af60ceb0acad1b2c513fc664c6dfadc9_2` | exp | 6 | 6 | 5 | 8 | Field tech is noisy; actual CSKH requirements match, but experience is above target. |
| 33 | `cskh_L4_4161488f5c40c84bf2e9f82b1e698e49_9` | exp | 6 | 7 | 4 | 8 | Strong operations/CSKH signals, but 6-year director is too senior for no-experience job. |
| 34 | `cskh_L4_4f15c0759a7c8958a3d20e01ea0c6215_13` | exp | 6 | 6 | 4 | 8 | CSKH/English/Excel fit, but seniority mismatch remains large. |
| 35 | `cskh_L4_90c81ec8e7e2d0fde869ddf09b995492_8` | exp | 7 | 8 | 4 | 7 | Very strong skill match, but no-experience job means overqualified penalty. |
| 36 | `cskh_L4_af60ceb0acad1b2c513fc664c6dfadc9_3` | exp | 6 | 6 | 4 | 8 | Relevant CSKH skills, but 6 years exceeds under-1-year job. |
| 37 | `cskh_L4_d9477e1b11710b99d38f8afffebd047d_4` | exp | 7 | 7 | 5 | 8 | Direct CSKH/English fit; 6 years is high for 1-year staff role. |
| 38 | `cskh_L4b_124f4256f0d6a862038089b366463a3e_6` | exp | 6 | 7 | 4 | 8 | Software-support CSKH skills fit, but 7-year director profile is too senior. |
| 39 | `cskh_L4b_526b6a5c136210bb26159bc9b387ce7c_12` | exp | 5 | 4 | 4 | 8 | Missing core Chinese HSK requirement; also overqualified. |
| 40 | `cskh_L4b_b41cb15e31e07b9c20c9f18fce0c36bf_13` | exp | 6 | 5 | 4 | 8 | Partial CSKH/office/English fit, missing Chinese HSK; overqualified. |

## Notes For Later Batches

This batch confirms the post-prompt issue to watch: models still often confuse "job không yêu cầu kinh nghiệm" with "any senior candidate gets 10". For final labels, overqualification should be treated consistently even when the candidate has strong domain skills.

