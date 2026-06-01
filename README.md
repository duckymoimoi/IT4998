# Hệ thống gợi ý việc làm cho CV tiếng Việt

Dự án xây dựng một ứng dụng gợi ý việc làm từ CV tiếng Việt. Hệ thống nhận CV của ứng viên, trích xuất thông tin chính, tìm các việc làm phù hợp trong Elasticsearch, sau đó sắp xếp lại nhóm kết quả tốt nhất bằng đánh giá đa chiều.

Repo public này tập trung vào **ứng dụng chạy được** và **bộ dữ liệu phục vụ demo/thử nghiệm**. Các script nội bộ dùng để chấm nhãn, adjudication và đánh giá thực nghiệm không được đưa vào repo public.

## Công nghệ sử dụng

- **Flask**: giao diện web và API backend.
- **Elasticsearch**: lưu trữ và truy vấn tin tuyển dụng.
- **BM25 + vector search + RRF**: tìm kiếm lai giữa khớp từ khóa và khớp ngữ nghĩa.
- **BGE-M3 / Sentence Transformers**: sinh embedding cho CV và job.
- **ESCO skill expansion**: mở rộng kỹ năng để giảm lệch cách diễn đạt giữa CV và JD.
- **Groq LLM API**: trích xuất CV và đánh giá mức phù hợp ở các chiều ngữ nghĩa.
- **Goong Maps API**: tùy chọn tính khoảng cách địa lý chi tiết; mặc định ưu tiên cùng tỉnh/thành.
- **Docker Compose**: chạy Elasticsearch và ứng dụng nhất quán trên máy local.

## Pipeline hệ thống

1. **Nhập CV**  
   Người dùng tải lên CV dạng PDF hoặc ảnh. Hệ thống trích xuất các trường chính như kỹ năng, kinh nghiệm, học vấn, vị trí mong muốn, địa điểm và lương kỳ vọng.

2. **Chuẩn hóa truy vấn**  
   Thông tin từ CV được gom thành truy vấn tìm kiếm. Các kỹ năng được mở rộng bằng ESCO để tăng khả năng tìm được JD có cùng ý nghĩa nhưng diễn đạt khác từ.

3. **Hybrid retrieval**  
   Elasticsearch trả về danh sách ứng viên bằng kết hợp BM25, vector search và RRF. Tầng này có nhiệm vụ thu hẹp không gian tìm kiếm từ toàn bộ dữ liệu việc làm xuống một tập kết quả có khả năng phù hợp cao.

4. **Ưu tiên địa điểm**  
   Nếu CV có địa điểm, hệ thống ưu tiên các công việc cùng tỉnh/thành trước khi gọi scoring chi tiết. Khoảng cách km theo Goong có thể được tính khi người dùng cần xem sâu từng job.

5. **Scoring đa chiều**  
   Top kết quả được đánh giá theo các chiều như relevance, skills, experience, education, salary và location. Điểm cuối được tổng hợp bằng trọng số để tạo thứ hạng hiển thị.

6. **Hiển thị kết quả**  
   Ứng dụng trả về danh sách job phù hợp, điểm tổng, điểm từng chiều, nhận xét ngắn và thông tin tuyển dụng.

## Dữ liệu đi kèm

Repo có thể chứa bộ dữ liệu job đã xử lý trong `data/`, gồm các file như:

- `topcv_balanced_1300.csv`: bộ job chính dùng để import vào Elasticsearch.
- `topcv_balanced_650.csv`, `topcv_balanced_650_final.csv`: bộ job nhỏ hơn dùng cho kiểm tra.
- `skills*.csv`, `data/esco/`: dữ liệu kỹ năng ESCO phục vụ mở rộng truy vấn.
- `evaluation_cvs_*.json`, `evaluation_pairs*.json`: dữ liệu CV-job phục vụ kiểm thử chất lượng nếu cần.

Các file local như API key, cache embedding, model weight, log và CV cá nhân đã được đưa vào `.gitignore`.

## Cài đặt

Yêu cầu:

- Docker Desktop.
- GPU NVIDIA là khuyến nghị nếu chạy embedding/LLM local nặng; vẫn có thể tùy chỉnh để chạy CPU nhưng sẽ chậm hơn.
- API key cho các dịch vụ được dùng trong app, tối thiểu là Groq nếu muốn dùng trích xuất CV và scoring bằng LLM.

Tạo file cấu hình local:

```powershell
Copy-Item src\.env.example src\.env
```

Điền các key cần dùng vào `src/.env`:

```env
GROQ_API_KEY=
GOONG_API_KEY=
ES_HOST=http://localhost:9200
ES_INDEX=topcv_jobs_1300
ENABLE_SKILL_GRAPH=0
ENABLE_CITY_PRIORITY=1
LOCATION_SCORE_MODE=city
```

Khởi động Elasticsearch:

```powershell
docker compose up -d elasticsearch
```

Import bộ job 1300 vào Elasticsearch:

```powershell
docker compose run --rm --entrypoint "" web python src/import_to_elastic.py --csv data/topcv_balanced_1300.csv --index topcv_jobs_1300 --es-host http://elasticsearch:9200
```

Chạy ứng dụng:

```powershell
docker compose up web
```

Mở trình duyệt tại:

```text
http://localhost:5000
```

## Lưu ý trước khi public

- Không commit `src/.env`.
- Không commit model weight, embedding cache, log, file PDF CV cá nhân hoặc output build LaTeX.
- Nếu API key từng nằm trong file local hoặc từng bị chia sẻ, nên rotate key trước khi public repo.
