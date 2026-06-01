# Data Directory

Thu muc nay chua du lieu da xu ly cho thuc nghiem CV-job matching.

## Du lieu nen commit

- `topcv_balanced_650.csv`, `topcv_balanced_650_final.csv`: bo job cu dung cho so sanh.
- `topcv_balanced_1300.csv`: bo job chinh dung de danh gia search.
- `evaluation_cvs_*.json`: CV sinh/tong hop dung trong danh gia.
- `evaluation_pairs*.json`: cap CV-job va nhan danh gia.
- `skills*.csv`: du lieu ESCO da tai va lam giau phuc vu expansion.
- `skill_profiles_1300.json`: profile ky nang da trich cho bo 1300.

## Du lieu khong nen commit

- `cache/`: embedding cache va artifact sinh tu dong.
- `*.pdf`: CV hoac tai lieu dau vao ca nhan.
- File log, file tam hoac output thu nghiem cuc bo.

Nhung muc khong nen commit da duoc khai bao trong `.gitignore`.
