# Data Directory

Thu muc nay chua du lieu da xu ly cho thuc nghiem CV-job matching.

## Du lieu nen commit

- `jobs/topcv_balanced_1300.csv`: bo job chinh dung cho import Elasticsearch.
- `turn1300_v3_union_final/`: bo du lieu thuc nghiem cuoi dung trong chuong 4.
- `search_quality_current/`: ket qua sweep va production-aligned duoc dua vao bao cao.
- `skills*.csv`: du lieu ESCO da tai va lam giau phuc vu expansion.
- `job_term_taxonomy*.json`, `job_terms_unique.csv`: bo tu vung noi bo va cac file kiem soat.

## Du lieu khong nen commit

- `cache/`: embedding cache va artifact sinh tu dong.
- `*.pdf`: CV hoac tai lieu dau vao ca nhan.
- File log, file tam hoac output thu nghiem cuc bo.

Nhung muc khong nen commit da duoc khai bao trong `.gitignore`.
