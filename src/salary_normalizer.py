
import re
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SalaryNormalizer:

    def __init__(self):
        # Conversion rates
        self.HOURS_PER_MONTH = 160  # ~8h/day * 20 days
        self.DAYS_PER_MONTH = 22
        self.USD_TO_VND = 24000

    def normalize_salary(self, salary_text: Union[str, int, float]) -> Dict:

        if isinstance(salary_text, (int, float)):
            return {
                'min': float(salary_text),
                'max': float(salary_text),
                'type': 'fixed',
                'original': str(salary_text)
            }

        if not salary_text or not isinstance(salary_text, str):
            return {
                'min': None,
                'max': None,
                'type': 'unknown',
                'original': str(salary_text)
            }

        text = salary_text.lower().strip()

        # Case 1: Negotiable
        negotiable_keywords = [
            'thoa thuan', 'thoả thuận', 'thỏa thuận',
            'trao doi', 'trao đổi', 'deal', 'negotiate', 'competitive'
        ]
        if any(word in text for word in negotiable_keywords):
            return {
                'min': None,
                'max': None,
                'type': 'negotiable',
                'original': salary_text
            }

        # Case 2: Hourly salary
        if any(word in text for word in ['gio', 'giờ', '/h', 'hour']):
            return self._parse_hourly_salary(text, salary_text)

        # Case 3: Daily salary
        if any(word in text for word in ['ngay', 'ngày', '/day', 'daily']):
            return self._parse_daily_salary(text, salary_text)

        # Case 4: USD salary
        if '$' in text or 'usd' in text:
            return self._parse_usd_salary(text, salary_text)

        # Case 5: Range (10-20 million, 10 - 20tr, etc.)
        range_match = re.search(r'(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)', text)
        if range_match:
            return self._parse_range_salary(text, range_match, salary_text)

        # Case 6: Fixed salary (Upto 20tr, From 15tr, Above 10tr)
        return self._parse_fixed_salary(text, salary_text)

    def _parse_hourly_salary(self, text: str, original: str) -> Dict:
        """Parse hourly salary"""
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        if not numbers:
            return self._error_result('hourly', original)

        hourly_rate = float(numbers[0])
        hourly_rate = self._apply_unit_multiplier(hourly_rate, text)
        monthly = hourly_rate * self.HOURS_PER_MONTH

        return {
            'min': monthly,
            'max': monthly,
            'type': 'hourly',
            'original': original
        }

    def _parse_daily_salary(self, text: str, original: str) -> Dict:
        """Parse daily salary"""
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        if not numbers:
            return self._error_result('daily', original)

        daily_rate = float(numbers[0])
        daily_rate = self._apply_unit_multiplier(daily_rate, text)
        monthly = daily_rate * self.DAYS_PER_MONTH

        return {
            'min': monthly,
            'max': monthly,
            'type': 'daily',
            'original': original
        }

    def _parse_usd_salary(self, text: str, original: str) -> Dict:
        """Parse USD salary"""
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        if not numbers:
            return self._error_result('usd', original)

        if len(numbers) >= 2 and any(sep in text for sep in ['-', '–', '—']):
            # Range
            min_val = float(numbers[0]) * self.USD_TO_VND
            max_val = float(numbers[1]) * self.USD_TO_VND
            return {
                'min': min_val,
                'max': max_val,
                'type': 'usd_range',
                'original': original
            }
        else:
            val = float(numbers[0]) * self.USD_TO_VND
            return {
                'min': val,
                'max': val,
                'type': 'usd',
                'original': original
            }

    def _parse_range_salary(self, text: str, match: re.Match, original: str) -> Dict:
        """Parse range salary"""
        min_val = float(match.group(1))
        max_val = float(match.group(2))

        # Apply unit multiplier
        min_val = self._apply_unit_multiplier(min_val, text)
        max_val = self._apply_unit_multiplier(max_val, text)

        return {
            'min': min_val,
            'max': max_val,
            'type': 'range',
            'original': original
        }

    def _parse_fixed_salary(self, text: str, original: str) -> Dict:
        """Parse fixed salary (Upto, From, Above...)"""
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        if not numbers:
            return self._error_result('unknown', original)

        val = float(numbers[0])
        val = self._apply_unit_multiplier(val, text)

        # Determine if it's min, max or fixed
        upto_keywords = ['upto', 'len den', 'lên đến', 'toi da', 'tối đa', 'max']
        from_keywords = ['tu', 'từ', 'from', 'tren', 'trên', 'above', 'over']

        if any(word in text for word in upto_keywords):
            return {
                'min': 0,
                'max': val,
                'type': 'upto',
                'original': original
            }
        elif any(word in text for word in from_keywords):
            return {
                'min': val,
                'max': None,
                'type': 'from',
                'original': original
            }
        else:
            return {
                'min': val,
                'max': val,
                'type': 'fixed',
                'original': original
            }

    def _apply_unit_multiplier(self, value: float, text: str) -> float:
        """Apply unit multiplier (thousand/million/billion)"""
        # Check for billion first (less common)
        if any(word in text for word in ['ty', 'tỷ', 'billion']):
            return value * 1_000_000_000

        # Check for million
        if any(word in text for word in ['tr', 'trieu', 'triệu', 'million']):
            return value * 1_000_000

        # Check for thousand
        if any(word in text for word in ['k', 'nghin', 'nghìn', 'thousand']):
            return value * 1_000

        # Default: assume it's already in VND
        return value

    def _error_result(self, salary_type: str, original: str) -> Dict:
        """Return error result"""
        return {
            'min': None,
            'max': None,
            'type': salary_type,
            'original': original
        }

    def compare_salary(
            self,
            expected_salary: Union[str, int, float],
            job_salary: Union[str, int, float, Dict]
    ) -> float:
        # Normalize both salaries
        expected = self.normalize_salary(expected_salary)

        # Handle dict input for job_salary
        if isinstance(job_salary, dict):
            job = job_salary
        else:
            job = self.normalize_salary(job_salary)

        # Case 1: Expected salary is negotiable → neutral score
        if expected['type'] == 'negotiable':
            return 70.0

        # Case 2: Job salary is negotiable → good score (flexible)
        if job['type'] == 'negotiable':
            return 75.0

        # Case 3: No valid salary data
        if expected['min'] is None or job['min'] is None:
            return 50.0

        # Get comparison values
        expected_val = expected.get('max') or expected.get('min', 0)
        job_min = job.get('min', 0)
        job_max = job.get('max') or job.get('min', 0)

        # Case 4: Expected salary within job range → perfect match
        if job_min <= expected_val <= job_max:
            return 100.0

        # Case 5: Job salary higher than expected → very good
        if job_min >= expected_val:
            return 95.0

        # Case 6: Job salary lower than expected
        # The closer to expected, the higher the score
        if job_max < expected_val:
            ratio = job_max / expected_val if expected_val > 0 else 0
            # Max 85 points if lower (not ideal but acceptable)
            return max(20.0, ratio * 85)

        # Default: partial match
        return 60.0
