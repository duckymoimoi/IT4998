
import os
import json
import re
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
DEFAULT_PARSE_SEED = _env_int("GROQ_PARSE_SEED", 42)
DEFAULT_PARSE_PROVIDER = os.environ.get("CV_PARSE_PROVIDER", "auto").strip().lower()
DEFAULT_COHERE_PARSE_MODEL = os.environ.get("COHERE_PARSE_MODEL", "command-r-08-2024")
DEFAULT_COHERE_PARSE_MAX_TOKENS = _env_int("COHERE_PARSE_MAX_TOKENS", 1800)

LANGUAGE_CERT_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"IELTS|TOEIC|TOEFL(?:\s*iBT|\s*ITP)?|JLPT|HSK|TOPIK|VSTEP|APTIS|"
    r"Cambridge|PET|KET|FCE|CAE|CPE"
    r")\b"
)
LANGUAGE_NAMES = [
    "Tiếng Anh", "Tiếng Nhật", "Tiếng Trung", "Tiếng Hàn", "Tiếng Pháp",
    "Tiếng Đức", "Tiếng Nga", "Tiếng Tây Ban Nha", "Tiếng Việt",
]


def _split_items(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,;\n]", str(value))
    return [re.sub(r"\s+", " ", str(item)).strip(" .:-") for item in raw_items if str(item).strip()]


def _dedupe(items: list[str]) -> list[str]:
    output, seen = [], set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item)).strip(" .:-")
        key = cleaned.lower()
        if cleaned and key not in seen:
            output.append(cleaned)
            seen.add(key)
    return output


def _language_name_from_text(text: str) -> str:
    for name in LANGUAGE_NAMES:
        if name.lower() in str(text).lower():
            return name
    return ""


def _extract_language_certificate(text: str) -> str:
    """Return a concise language-certificate label if text contains one."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return ""

    patterns = [
        r"(?i)\bIELTS\s*\d+(?:[.,]\d+)?\+?\b",
        r"(?i)\bTOEIC\s*\d+\+?\b",
        r"(?i)\bTOEFL(?:\s*iBT|\s*ITP)?\s*\d+\+?\b",
        r"(?i)\bJLPT\s*N?[1-5]\b",
        r"(?i)\bHSK\s*[1-6]\b",
        r"(?i)\bTOPIK\s*[1-6]\b",
        r"(?i)\bVSTEP\s*[A-C][1-2]\b",
        r"(?i)\bAPTIS\s*[A-C][1-2]\b",
        r"(?i)\b(?:PET|KET|FCE|CAE|CPE)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()

    if re.search(r"(?i)\btiếng\s+nhật\b", value):
        match = re.search(r"(?i)\bN[1-5]\b", value)
        if match:
            return f"JLPT {match.group(0).upper()}"
    if re.search(r"(?i)\btiếng\s+trung\b", value):
        match = re.search(r"(?i)\bHSK\s*[1-6]\b", value)
        if match:
            return re.sub(r"\s+", "", match.group(0).upper())
    return ""


def normalize_language_certificate_fields(data: Dict) -> Dict:
    """Move language certificates out of languages and into certificates."""
    languages = []
    certificates = _split_items(data.get("certificates", ""))

    for item in _split_items(data.get("languages", "")):
        cert = _extract_language_certificate(item)
        lang_name = _language_name_from_text(item)
        if cert:
            certificates.append(cert)
            if lang_name:
                languages.append(lang_name)
        else:
            languages.append(item)

    cleaned_certs = []
    for item in certificates:
        cert = _extract_language_certificate(item)
        if cert:
            cleaned_certs.append(cert)
            lang_name = _language_name_from_text(item)
            if lang_name:
                languages.append(lang_name)
        elif _language_name_from_text(item) and not LANGUAGE_CERT_PATTERN.search(item):
            languages.append(item)
        else:
            cleaned_certs.append(item)

    data["languages"] = ", ".join(_dedupe(languages))
    data["certificates"] = ", ".join(_dedupe(cleaned_certs))
    return data


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

**PHÂN LOẠI KỸ NĂNG (chia thành 3 nhóm riêng biệt):**
1. technical_skills: Ngôn ngữ lập trình, framework, database, tools, phần mềm chuyên ngành
   Ví dụ: Python, React, MySQL, Docker, Git, Excel, AutoCAD, SAP, Photoshop, Figma
2. languages: Ngoại ngữ và trình độ
   Ví dụ: Tiếng Anh B2, Tiếng Nhật, Tiếng Trung
   KHÔNG đưa chứng chỉ/điểm thi ngoại ngữ vào languages.
3. certificates: Chứng chỉ chuyên môn và chứng chỉ ngoại ngữ
   Ví dụ: IELTS 7.0, TOEIC 800, TOEFL iBT 90, JLPT N3, HSK4, AWS Solutions Architect, PMP, CPA, CCNA

**CV TEXT:**
{cv_text[:10000]}

**Trả về JSON (KHÔNG có markdown backticks):**
{{
    "target_roles": "Data Scientist, Data Analyst",
    "core_skills": "Python, Scikit-learn, PyTorch, Pandas, SQL",
    "secondary_skills": "Hadoop, Spark, Docker",
    "technical_skills": "Python, Scikit-learn, PyTorch, React, MySQL, Docker, Git",
    "languages": "Tiếng Anh B2",
    "certificates": "IELTS 7.0, TOEIC 800, AWS Solutions Architect",
    "experience": "3",
    "education": "dai_hoc",
    "gender": "Nam",
    "location": "Hồ Chí Minh",
}}

**RULES:**
- target_roles: Tối đa 3 vai trò nghề nghiệp có bằng chứng rõ từ tiêu đề CV, kinh nghiệm hoặc dự án. Không tự suy diễn.
- core_skills: Tối đa 12 kỹ năng trực tiếp chứng minh ứng viên làm được target_roles. Ưu tiên kỹ năng xuất hiện trong kinh nghiệm, nhiều dự án liên quan hoặc được dùng để tạo kết quả cụ thể.
- secondary_skills: Tối đa 8 kỹ năng có quan hệ trực tiếp với công việc của target_roles nhưng bằng chứng yếu hơn hoặc chỉ hỗ trợ core_skills. Loại kỹ năng chỉ xuất hiện trong dự án lệch target_roles, kể cả khi đó là kỹ năng kỹ thuật hợp lệ. Không lặp core_skills.
- Sắp xếp core_skills và secondary_skills theo độ liên quan và độ mạnh bằng chứng giảm dần. Khi không chắc một kỹ năng có hỗ trợ target_roles hay không, chỉ giữ nó trong technical_skills; secondary_skills có thể rỗng.
- technical_skills: Toàn bộ kỹ năng kỹ thuật có bằng chứng trong CV. Kỹ năng lệch target_roles vẫn có thể nằm ở đây nhưng không được ép vào core_skills/secondary_skills.
- languages: chỉ ghi tên ngoại ngữ hoặc trình độ ngôn ngữ không gắn với một chứng chỉ cụ thể, ví dụ "Tiếng Anh B2", "Tiếng Nhật".
- certificates: tất cả chứng chỉ/điểm thi ngoại ngữ như IELTS, TOEIC, TOEFL, JLPT, HSK, TOPIK, VSTEP, APTIS phải nằm ở certificates, không nằm ở languages.
- experience: Số năm làm việc (1, 2, 3, 4, 5 hoặc "over_5"). Nếu fresher/sinh viên → "under_1"
- education: "dai_hoc" (đại học), "cao_dang", "trung_cap", "trung_hoc"
- gender: "Nam", "Nữ", hoặc "both" (nếu không rõ)
- location: Tên thành phố Việt Nam chính xác (ví dụ: "Hồ Chí Minh", "Hà Nội", "Đà Nẵng")
- Mỗi nhóm kỹ năng là chuỗi phân tách bằng dấu phẩy. Nếu không tìm thấy → chuỗi rỗng ""
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
                response_format={"type": "json_object"},
                seed=DEFAULT_PARSE_SEED,
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
{cv_text[:10000]}

JSON schema:
{{
  "target_roles": "Data Scientist, Data Analyst",
  "core_skills": "Python, Scikit-learn, Pandas, SQL",
  "secondary_skills": "Hadoop, Spark, Docker",
  "technical_skills": "Python, React, SQL",
  "languages": "Tiếng Anh B2",
  "certificates": "IELTS 7.0, TOEIC 800, AWS, PMP",
  "experience": "under_1",
  "education": "dai_hoc",
  "gender": "Nam",
  "location": "Hà Nội",
}}

Rules:
- target_roles: tối đa 3 vai trò có bằng chứng rõ từ tiêu đề CV, kinh nghiệm hoặc dự án.
- core_skills: tối đa 12 kỹ năng trực tiếp chứng minh ứng viên làm được target_roles.
- secondary_skills: tối đa 8 kỹ năng có quan hệ trực tiếp với công việc của target_roles nhưng bằng chứng yếu hơn hoặc chỉ hỗ trợ core_skills; loại kỹ năng chỉ xuất hiện trong dự án lệch target_roles và không lặp core_skills.
- Sắp xếp core_skills và secondary_skills theo độ liên quan và độ mạnh bằng chứng giảm dần. Khi không chắc, chỉ giữ kỹ năng trong technical_skills; secondary_skills có thể rỗng.
- technical_skills: toàn bộ ngôn ngữ lập trình, framework, database, tool và phần mềm chuyên ngành; kỹ năng lệch target_roles chỉ nằm ở đây.
- languages: chỉ ghi tên ngoại ngữ hoặc trình độ không gắn với chứng chỉ cụ thể, ví dụ "Tiếng Anh B2", "Tiếng Nhật".
- certificates: ghi toàn bộ chứng chỉ/điểm thi ngoại ngữ như IELTS, TOEIC, TOEFL, JLPT, HSK, TOPIK, VSTEP, APTIS; không để các mục này trong languages.
- languages/certificates: nếu không có thì chuỗi rỗng.
- experience: một trong "under_1", "1", "2", "3", "4", "5", "over_5".
- education: một trong "dai_hoc", "cao_dang", "trung_cap", "trung_hoc".
- gender: "Nam", "Nữ", hoặc "both" nếu không rõ.
- location: thành phố/tỉnh ở Việt Nam nếu có.
"""

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
                "target_roles": "",
                "skills": "",
                "experience": "",
                "education": "",
                "gender": "both",
                "location": "",
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
                "target_roles": "",
                "skills": "",
                "experience": "",
                "education": "",
                "gender": "both",
                "location": "",
            }

        groq_result = normalize_language_certificate_fields(groq_result)

        result = {
            "success": True,
            "target_roles": groq_result.get("target_roles", ""),
            "core_skills": groq_result.get("core_skills", ""),
            "secondary_skills": groq_result.get("secondary_skills", ""),
            "technical_skills": groq_result.get("technical_skills", ""),
            "languages": groq_result.get("languages", ""),
            "certificates": groq_result.get("certificates", ""),
            "experience": groq_result.get("experience", ""),
            "education": groq_result.get("education", ""),
            "gender": groq_result.get("gender", "both"),
            "location": groq_result.get("location", ""),
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
