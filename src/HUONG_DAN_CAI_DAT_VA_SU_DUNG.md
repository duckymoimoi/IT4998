# Hướng dẫn cài đặt và sử dụng

Tài liệu này mô tả cách chạy ứng dụng gợi ý việc làm, khởi tạo dữ liệu
Elasticsearch và thu thập tin tuyển dụng. Các lệnh được viết cho PowerShell tại
thư mục gốc của dự án.

## 1. Thành phần hệ thống

Các dịch vụ chính trong `docker-compose.yml`:

| Dịch vụ | Cổng | Mục đích |
|---|---:|---|
| `elasticsearch` | 9200 | Lưu trữ và truy xuất tin tuyển dụng |
| `web` | 5000 | Giao diện Flask, truy xuất lai và chấm điểm |
| `ollama` | 11434 | Mô hình cục bộ dùng khi làm sạch dữ liệu crawl |
| `scheduler` | Không mở cổng | Crawl, làm sạch và cập nhật Elasticsearch |

`ollama` và `scheduler` thuộc profile `crawl`, nên không tự chạy khi chỉ khởi
động ứng dụng web.

## 2. Yêu cầu

- Windows 10 hoặc 11 với Docker Desktop.
- Tối thiểu 8 GB RAM trống. Lần chạy đầu cần Internet để tải image và BGE-M3.
- Một khóa Groq cho bước phân tích CV và chấm nhanh.
- Khóa Goong chỉ cần khi tính khoảng cách di chuyển chi tiết.
- GPU NVIDIA không bắt buộc. Sinh embedding trên CPU sẽ chậm hơn.

Kiểm tra Docker:

```powershell
docker version
docker compose version
```

## 3. Tạo cấu hình

Gói này đã có sẵn tệp `src\.env` (tạo từ `src\.env.example`). Chỉ cần mở và điền
**ít nhất một khóa Groq** trước khi chạy:

```env
GROQ_API_KEY_1=YOUR_KEY
ES_INDEX=topcv_jobs_production
```

Có thể khai báo nhiều khóa từ `GROQ_API_KEY_1` đến `GROQ_API_KEY_k`. Nếu vô tình
xóa mất `src\.env`, hãy tạo lại:

```powershell
Copy-Item src\.env.example src\.env
```

Không đưa `src\.env` (chứa khóa thật) lên Git.

Các cấu hình vận hành quan trọng:

```env
RETRIEVAL_SIZE=100
RETRIEVAL_FALLBACK_MIN_SCORE=1.0
RETRIEVAL_FALLBACK_MAX_SCORE=5.0
SCORING_TOP_N=50
SCORING_BATCH_SIZE=5
BM25_QUERY_TERM_LIMIT=30
ENABLE_ESCO_EXPANSION=0
ENABLE_SKILL_GRAPH_EXPANSION=0
LOCATION_SCORE_MODE=city
LOCATION_DISTANCE_DECAY_KM=30
LOCATION_SCORE_FLOOR=1.5
```

> **Lưu ý về bản gọn:** Bản đóng gói này đã **tắt mở rộng truy vấn ESCO**
> (`ENABLE_ESCO_EXPANSION=0`) để loại bỏ ~52 MB cache embedding (`data/cache/*.npy`),
> giúp file nén dưới 30 MB. Truy xuất lai BM25 + kNN (BGE-M3) vẫn hoạt động đầy đủ;
> chỉ thiếu bước bổ sung kỹ năng theo từ điển ESCO nên chất lượng có thể giảm nhẹ.
>
> Muốn bật lại ESCO: chỉ cần đặt `ENABLE_ESCO_EXPANSION=1`. File từ điển
> `data/skills_with_names.csv` đã được kèm sẵn, nên lần chạy web đầu tiên sẽ **tự sinh
> lại cache embedding** (mất vài phút trên CPU) và lưu vào volume cho các lần sau.

## 4. Khởi động ứng dụng

Khởi động Elasticsearch:

```powershell
docker compose up -d elasticsearch
```

Kiểm tra trạng thái:

```powershell
docker compose ps
Invoke-RestMethod http://localhost:9200/_cluster/health
```

Nếu chỉ mục production đã có dữ liệu, khởi động web:

```powershell
docker compose up -d web
```

Mở `http://localhost:5000`.

Xem log khi ứng dụng khởi động lâu:

```powershell
docker compose logs -f web
```

Lần đầu BGE-M3 được tải vào volume cache nên có thể mất vài phút.

## 5. Khởi tạo chỉ mục từ CSV

Lệnh sau nhập bộ dữ liệu có sẵn và sinh embedding:

```powershell
docker compose run --rm --entrypoint "" web `
  python -m job_matching.ingestion.import_to_elastic `
  --csv data/jobs/topcv_balanced_1300.csv `
  --index topcv_jobs_production `
  --es-host http://elasticsearch:9200
```

Chỉ dùng `--no-embedding` khi cần kiểm tra BM25. Không có embedding thì nhánh
kNN và truy xuất lai không hoạt động đầy đủ.

Kiểm tra số tài liệu:

```powershell
Invoke-RestMethod http://localhost:9200/topcv_jobs_production/_count
```

Web không tự ghi đè chỉ mục khi khởi động vì `AUTO_IMPORT_ON_START=0`.

## 6. Sử dụng giao diện

1. Mở `http://localhost:5000`.
2. Tải CV dạng PDF hoặc ảnh, dung lượng tối đa 16 MB.
3. Kiểm tra các trường vai trò, kỹ năng, kinh nghiệm và học vấn đã trích xuất.
4. Nhập địa điểm và mức lương mong muốn khi cần.
5. Chọn **Tìm việc phù hợp**.
6. Tab **AI chấm nhanh** chứa nhóm đã được chấm đa tiêu chí.
7. Tab **Kết quả còn lại** giữ các công việc theo thứ hạng truy xuất.
8. Chọn **Phân tích chi tiết** để xem bằng chứng và điểm của một công việc.
9. Chọn **Khoảng cách** sau khi nhập địa chỉ cụ thể để cập nhật điểm địa điểm.

## 7. Crawl tin tuyển dụng

Khởi động Ollama và tải mô hình làm sạch:

```powershell
docker compose --profile crawl up -d ollama
docker exec do_an_ollama ollama pull qwen3.5:4b
```

Chạy một lượt mặc định gồm ba trang và ba luồng:

```powershell
docker compose --profile crawl run --rm scheduler
```

Chạy với tham số riêng:

```powershell
docker compose --profile crawl run --rm scheduler `
  python -m job_matching.ingestion.scheduler `
  --once --pages 5 --threads 2
```

Chỉ crawl và ghi CSV, không cập nhật Elasticsearch:

```powershell
docker compose --profile crawl run --rm scheduler `
  python -m job_matching.ingestion.scheduler `
  --once --pages 3 --threads 1 --no-es
```

Ép kiểm tra lại cả URL đã crawl gần đây:

```powershell
docker compose --profile crawl run --rm scheduler `
  python -m job_matching.ingestion.scheduler `
  --once --pages 3 --threads 1 --force-recrawl-existing
```

Cập nhật Elasticsearch từ một CSV hiện có:

```powershell
docker compose --profile crawl run --rm scheduler `
  python -m job_matching.ingestion.scheduler `
  --upsert-file data/jobs/topcv_pipeline_YYYYMMDD_HHMMSS.csv `
  --es-host http://elasticsearch:9200
```

Kết quả crawl nằm trong `data/jobs`. Log scheduler nằm trong `logs`.
