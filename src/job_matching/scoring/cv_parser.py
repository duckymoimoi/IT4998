
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


class CVParser:

    def __init__(self, groq_api_key: str = None):
        """
        Args:
            groq_api_key: Groq API key
        """
        self.api_key = groq_api_key or os.environ.get('GROQ_API_KEY') or os.environ.get('GROQ_API_KEY_1')

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Set environment variable or pass api_key parameter. "
                "Get free key at: https://console.groq.com/keys"
            )

        self.client = Groq(api_key=self.api_key)
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
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=2000,
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
        groq_result = self.analyze_with_groq(raw_text)

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
            "raw_text": raw_text[:3000],
            "extraction_method": "markitdown + groq_llama_3.3"
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
