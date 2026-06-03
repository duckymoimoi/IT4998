
import os
import json
from typing import Dict
from pathlib import Path

from groq import Groq
from markitdown import MarkItDown

# Job categories
JOB_CATEGORIES = [
    "Nhân viên kinh doanh",
    "Kế toán",
    "Marketing",
    "Hành chính nhân sự",
    "Chăm sóc khách hàng",
    "Ngân hàng",
    "IT",
    "Kỹ sư xây dựng",
    "Thiết kế đồ họa",
    "Bất động sản",
    "Giáo dục",
    "Telesales",
    "Lao động phổ thông",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DEFAULT_PARSE_MODEL = os.environ.get("GROQ_PARSE_MODEL", "llama-3.3-70b-versatile")
DEFAULT_PARSE_MAX_COMPLETION_TOKENS = _env_int("GROQ_PARSE_MAX_COMPLETION_TOKENS", 2000)
DEFAULT_PARSE_PROVIDER = os.environ.get("CV_PARSE_PROVIDER", "auto").strip().lower()
DEFAULT_COHERE_PARSE_MODEL = os.environ.get("COHERE_PARSE_MODEL", "command-r-08-2024")
DEFAULT_COHERE_PARSE_MAX_TOKENS = _env_int("COHERE_PARSE_MAX_TOKENS", 1800)


def _load_env_keys(prefix: str, max_index: int = 9):
    keys = []
    for i in range(1, max_index + 1):
        key = os.environ.get(f"{prefix}_{i}")
        if key:
            keys.append(key)
    single = os.environ.get(prefix)
    if single and single not in keys:
        keys.append(single)
    return keys


class CVParser:

    def __init__(self, groq_api_key: str = None):
        """
        Args:
            groq_api_key: Groq API key
        """
        self.api_key = groq_api_key or os.environ.get('GROQ_API_KEY') or os.environ.get('GROQ_API_KEY_1')
        self.cohere_api_keys = _load_env_keys("COHERE_API_KEY")

        if not self.api_key and not self.cohere_api_keys:
            raise ValueError(
                "No LLM API key found. Set GROQ_API_KEY or COHERE_API_KEY."
            )

        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.job_categories = JOB_CATEGORIES
        self._md_converter = MarkItDown()

    def extract_text(self, pdf_path: str) -> str:
        """Convert PDF/DOCX to Markdown using Microsoft MarkItDown.

        MarkItDown preserves document structure (headings, tables, lists)
        which significantly improves LLM extraction accuracy compared
        to raw text extraction.

        Args:
            pdf_path: Path to PDF or DOCX file

        Returns:
            Markdown-formatted text content
        """
        try:
            print(f"\n Converting to Markdown: {os.path.basename(pdf_path)}")
            result = self._md_converter.convert(pdf_path)
            text = result.text_content.strip()

            if len(text) > 100:
                print(f"  [OK] MarkItDown: {len(text)} chars extracted")
                return text
            else:
                print(f"  [WARN] MarkItDown returned too little text ({len(text)} chars)")
                return ""

        except Exception as e:
            print(f"  MarkItDown extraction failed: {e}")
            return ""

    def analyze_with_groq(self, cv_text: str) -> Dict:
        """
        Args:
            cv_text: Raw CV text content

        Returns:
            Dictionary with parsed CV data
        """

        prompt = f"""Bạn là chuyên gia phân tích CV tiếng Việt. Trích xuất thông tin chi tiết từ CV.

**PHÂN LOẠI KỸ NĂNG (chia thành 4 nhóm riêng biệt):**
1. technical_skills: Ngôn ngữ lập trình, framework, database, tools, phần mềm chuyên ngành
   Ví dụ: Python, React, MySQL, Docker, Git, Excel, AutoCAD, SAP, Photoshop, Figma
2. soft_skills: Kỹ năng mềm
   Ví dụ: Giao tiếp, Teamwork, Quản lý thời gian, Lãnh đạo, Giải quyết vấn đề
3. languages: Ngoại ngữ và trình độ
   Ví dụ: Tiếng Anh B2, TOEIC 850, Tiếng Nhật N3, Tiếng Trung HSK4
4. certificates: Chứng chỉ chuyên môn
   Ví dụ: AWS Solutions Architect, TOEIC 800, PMP, CPA, CCNA

**CV TEXT:**
{cv_text[:4000]}

**Trả về JSON (KHÔNG có markdown backticks):**
{{
    "technical_skills": "Python, Scikit-learn, PyTorch, React, MySQL, Docker, Git",
    "soft_skills": "Giao tiếp, Teamwork, Quản lý thời gian",
    "languages": "Tiếng Anh B2, TOEIC 800",
    "certificates": "AWS Solutions Architect, TOEIC 800",
    "experience": "3",
    "education": "dai_hoc",
    "gender": "Nam",
    "location": "Hồ Chí Minh",
    "suggested_categories": ["IT"]
}}

**RULES:**
- experience: Số năm làm việc (1, 2, 3, 4, 5 hoặc "over_5"). Nếu fresher/sinh viên → "under_1"
- education: "dai_hoc" (đại học), "cao_dang", "trung_cap", "trung_hoc"
- gender: "Nam", "Nữ", hoặc "both" (nếu không rõ)
- location: Tên thành phố Việt Nam chính xác (ví dụ: "Hồ Chí Minh", "Hà Nội", "Đà Nẵng")
- Mỗi nhóm kỹ năng là chuỗi phân tách bằng dấu phẩy. Nếu không tìm thấy → chuỗi rỗng ""
- suggested_categories: Chọn từ danh sách: {json.dumps(self.job_categories, ensure_ascii=False)}
"""

        try:

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia phân tích CV. Luôn trả về valid JSON without markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=DEFAULT_PARSE_MODEL,
                temperature=0.1,
                max_completion_tokens=DEFAULT_PARSE_MAX_COMPLETION_TOKENS,
            )

            response_text = chat_completion.choices[0].message.content.strip()

            # Clean markdown backticks
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()

            # Parse JSON
            cv_data = json.loads(response_text)

            return cv_data

        except json.JSONDecodeError as e:
            print(f" JSON parsing error: {e}")
            print(f"Response preview: {response_text[:500]}\n")
            return None
        except Exception as e:
            print(f" Groq analysis failed: {e}\n")
            import traceback
            traceback.print_exc()
            return None

    def analyze_with_cohere(self, cv_text: str) -> Dict:
        """Analyze CV with Cohere Chat v2 REST. Used when Groq quota is exhausted."""
        if not self.cohere_api_keys:
            return None

        prompt = f"""Bạn là chuyên gia phân tích CV tiếng Việt. Trích xuất thông tin từ CV sau và chỉ trả về JSON hợp lệ, không thêm markdown.

CV TEXT:
{cv_text[:4500]}

JSON schema:
{{
  "technical_skills": "Python, React, SQL",
  "soft_skills": "Giao tiếp, Teamwork",
  "languages": "Tiếng Anh B2, TOEIC 800",
  "certificates": "AWS, PMP",
  "experience": "under_1",
  "education": "dai_hoc",
  "gender": "Nam",
  "location": "Hà Nội",
  "suggested_categories": ["IT"]
}}

Rules:
- technical_skills: ngôn ngữ lập trình, framework, database, tool, phần mềm chuyên ngành.
- soft_skills/languages/certificates: nếu không có thì chuỗi rỗng.
- experience: một trong "under_1", "1", "2", "3", "4", "5", "over_5".
- education: một trong "dai_hoc", "cao_dang", "trung_cap", "trung_hoc".
- gender: "Nam", "Nữ", hoặc "both" nếu không rõ.
- location: thành phố/tỉnh ở Việt Nam nếu có.
- suggested_categories: chọn từ danh sách {json.dumps(self.job_categories, ensure_ascii=False)}."""

        for idx, api_key in enumerate(self.cohere_api_keys):
            try:
                import requests
                response = requests.post(
                    "https://api.cohere.com/v2/chat",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": DEFAULT_COHERE_PARSE_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Bạn là chuyên gia phân tích CV. Chỉ trả về valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": DEFAULT_COHERE_PARSE_MAX_TOKENS,
                    },
                    timeout=90,
                )
                if response.status_code == 429:
                    print(f" Cohere parse key {idx + 1} rate limited, trying next key...")
                    continue
                if response.status_code >= 400:
                    print(f" Cohere parse error {response.status_code}: {response.text[:200]}")
                    continue

                data = response.json()
                content = data.get("message", {}).get("content", [])
                response_text = ""
                if content and isinstance(content, list):
                    response_text = content[0].get("text", "")
                if not response_text:
                    response_text = data.get("text", "")

                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "").strip()

                start = response_text.find("{")
                end = response_text.rfind("}")
                if start != -1 and end > start:
                    response_text = response_text[start:end + 1]
                return json.loads(response_text)

            except json.JSONDecodeError as e:
                print(f" Cohere JSON parsing error: {e}")
            except Exception as e:
                print(f" Cohere analysis failed on key {idx + 1}: {str(e)[:200]}")

        return None

    def parse_cv(self, file_path: str) -> Dict:

        raw_text = self.extract_text(file_path)

        if not raw_text or len(raw_text.strip()) < 50:
            return {
                "success": False,
                "error": "Không thể đọc nội dung từ CV. File có thể bị mã hóa hoặc là ảnh scan.",
                "raw_text": raw_text[:500] if raw_text else "",
                "skills": "",
                "experience": "",
                "education": "",
                "gender": "both",
                "location": "",
                "suggested_categories": [],
            }
        groq_result = None
        if DEFAULT_PARSE_PROVIDER in {"groq", "auto"}:
            groq_result = self.analyze_with_groq(raw_text)
        if not groq_result and DEFAULT_PARSE_PROVIDER in {"cohere", "auto"}:
            groq_result = self.analyze_with_cohere(raw_text)

        if not groq_result:
            return {
                "success": False,
                "error": "Groq AI không thể phân tích CV. Vui lòng thử lại.",
                "raw_text": raw_text[:1000],
                "skills": "",
                "experience": "",
                "education": "",
                "gender": "both",
                "location": "",
                "suggested_categories": [],
            }

        result = {
            "success": True,
            "technical_skills": groq_result.get("technical_skills", ""),
            "soft_skills": groq_result.get("soft_skills", ""),
            "languages": groq_result.get("languages", ""),
            "certificates": groq_result.get("certificates", ""),
            "experience": groq_result.get("experience", ""),
            "education": groq_result.get("education", ""),
            "gender": groq_result.get("gender", "both"),
            "location": groq_result.get("location", ""),
            "suggested_categories": groq_result.get("suggested_categories", []),
            "raw_text": raw_text[:8000],
            "extraction_method": f"markitdown + {DEFAULT_PARSE_PROVIDER}"
        }
        return result


_parser_instance = None


def get_cv_parser(groq_api_key: str = None) -> CVParser:
    """Get or create CVParser singleton"""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = CVParser(groq_api_key)
    return _parser_instance


def parse_cv_file(file_path: str, groq_api_key: str = None) -> Dict:

    parser = get_cv_parser(groq_api_key)
    return parser.parse_cv(file_path)
