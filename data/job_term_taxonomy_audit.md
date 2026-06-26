# Job Term Taxonomy Audit

## Summary

```json
{
  "unique_terms_csv": 1246,
  "taxonomy_rows": 1246,
  "missing_from_taxonomy": 0,
  "extra_in_taxonomy": 0,
  "type_counts": {
    "tool": 309,
    "technical_skill": 148,
    "sales_channel": 21,
    "role": 143,
    "domain": 97,
    "professional_skill": 365,
    "language": 30,
    "soft_skill": 72,
    "noise": 50,
    "certification": 11
  },
  "weird_key_count": 0,
  "invalid_count": 0,
  "low_conf_count": 0,
  "noise_review_remaining": 6
}
```

## Remaining noise terms reviewed
- `máy tính` -> noise (Quá chung, không đủ đặc trưng để search)
- `Theo dõi` -> noise (Quá chung, không đủ đặc trưng để search)
- `phụ kiện` -> noise (Quá chung, thiếu domain cụ thể)
- `Tốt nghiệp Đại học chuyên ngành Cơ Điện Lạnh / MEP` -> noise (Thông tin bằng cấp, không phải kỹ năng)
- `FAST FRO` -> noise (Không đủ ngữ cảnh, không xác định được tool)
- `live PC` -> noise (Không đủ ngữ cảnh để xác định)

## Notes

- Taxonomy contains one row per unique `technical_skills` term.
- Overrides are stored in `data/job_term_taxonomy_overrides.json` and can be re-applied with `--apply-overrides`.
- Remaining noise terms above are intentionally kept out of query terms because they are too broad, education-only, or lack context.