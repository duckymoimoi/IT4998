"""
Enrichment script cho ESCO skills.csv
- Trich xuat ten skill (preferred label) tu description
- Tach description thanh: preferred_label, alt_labels, description
- Xuat file skills_enriched.csv voi day du cot

Logic: Trong ESCO CSV, cot 'description' co format:
  "preferred_label alt_label_1 alt_label_2 ... Description sentence."
  
  VD: "Haskell Haskell techniques The techniques and principles of software development..."
  => preferred_label = "Haskell"
  => description_text = "The techniques and principles of software development..."
  
  VD: "manage musical staff manage staff of music ... Assign and manage staff tasks..."
  => preferred_label = "manage musical staff"
  => description_text = "Assign and manage staff tasks..."

Heuristic: Cau mo ta bat dau bang chu in hoa va ket thuc bang dau cham.
Phan truoc cau mo ta chinh la cac labels.
"""

import pandas as pd
import re
import sys
import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


def extract_preferred_label_heuristic(description):
    """
    Trich xuat preferred label tu description.
    
    Heuristic:
    1. Tim cau dau tien bat dau bang chu in hoa (A-Z) co do dai > 20 chars
       => Do la bat dau cua description text
    2. Phan text truoc do la cac labels
    3. Label dau tien (truoc dau cach hoac pattern lap lai) la preferred label
    """
    if not description or not isinstance(description, str):
        return "", "", ""
    
    text = description.strip()
    
    # Tim vi tri bat dau cua description (cau dau tien voi chu in hoa + dai)
    # Pattern: Bat dau tu mot chu in hoa, tiep theo la tu + dau cham hoac dau phay
    # Mo ta thuong bat dau = "The ...", "A ...", "An ...", etc.
    
    desc_start_patterns = [
        # "The techniques and principles..."
        r'(?:^|\s)(The\s+[a-z])',
        # "A plan-based business..."
        r'(?:^|\s)(A\s+[a-z])',
        # "An integrated approach..."
        r'(?:^|\s)(An\s+[a-z])',
        # "Various techniques..."
        r'(?:^|\s)(Various\s+[a-z])',
        # Bat dau bang dong tu + object (imperative): "Assign and manage...", "Identify oppression..."
        # => Tim cau dai bat dau bang chu in hoa
        # General: Sentence starting with uppercase, followed by lowercase words
    ]
    
    # Strategy 1: Tim sentence bat dau bang capital letter co van phong mo ta
    # Mo ta thuong o cuoi, sau tat ca labels
    
    # Cach tiep can tot hon: labels thuong la cac cum ngan, lap lai y nghia
    # Mo ta la cau dai, day du, thuong chua dau cham
    
    # Tim tat ca vi tri ma mot sentence moi bat dau (chu in hoa sau khoang trang)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 1:
        # Khong co dau cham => toan bo la labels, label dau la preferred
        words = text.split()
        # Tim nhom tu dau tien = preferred label
        # Labels thuong phan cach boi cung pattern lap lai
        preferred = _extract_first_label(text)
        return preferred, "", text
    
    # Tim cau cuoi cung du dai -> do la description
    # Labels nam truoc description
    desc_text = ""
    labels_text = text
    
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
        # Kiem tra: cau nay co phai la mo ta dai khong?
        # Description thuong > 40 chars va bat dau bang chu in hoa
        if len(sent) > 40 and sent[0].isupper():
            # Kiem tra khong phai la label (labels thuong ngan va khong co dau cham)
            if '.' in sent or len(sent) > 80:
                desc_text = '. '.join(sentences[i:])
                labels_text = '. '.join(sentences[:i]) if i > 0 else ""
                break
    
    if not desc_text:
        # Fallback: sentence cuoi = description
        desc_text = sentences[-1] if len(sentences) > 1 else ""
        labels_text = '. '.join(sentences[:-1]) if len(sentences) > 1 else text
    
    # Extract preferred label (label dau tien trong labels_text)
    preferred = _extract_first_label(labels_text)
    
    return preferred, labels_text, desc_text


def _extract_first_label(labels_text):
    """
    Trich xuat preferred label (label dau tien) tu chuoi labels.
    
    Labels co pattern: "label1 label2 label3 ..."
    Moi label la mot cum tu.
    
    Ve co ban, preferred label = cum tu dau tien truoc khi pattern bat dau lap lai.
    """
    if not labels_text:
        return ""
    
    text = labels_text.strip()
    words = text.split()
    
    if not words:
        return ""
    
    # Truong hop don gian: chi co 1-3 tu => toan bo la label
    if len(words) <= 3:
        return text
    
    # Tim preferred label = cum tu dau tien
    # Heuristic: Preferred label ket thuc khi:
    # 1. Gap tu bat dau bang chu in hoa (label moi)
    # 2. Gap pattern lap lai (tu da xuat hien)
    
    # Cach 1: Tim cum dau tien truoc khi cam thay la label moi
    seen_words = set()
    label_end = len(words)
    
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,;:')
        if word_lower in seen_words and i > 0:
            label_end = i
            break
        seen_words.add(word_lower)
        
        # Neu gap cum > 5 tu ma chua co lap lai, lay 3-4 tu dau
        if i >= 5:
            label_end = min(4, len(words))
            break
    
    return ' '.join(words[:label_end])


def enrich_esco_csv(input_path, output_path):
    """
    Doc skills.csv goc va tao file enriched.
    """
    print(f"Doc file: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Tong so skills: {len(df)}")
    
    # Extract URI ID
    df['skill_id'] = df['id'].apply(lambda x: str(x).split('/')[-1] if pd.notna(x) else "")
    
    # Extract preferred label, labels, description
    results = df['description'].apply(extract_preferred_label_heuristic)
    df['preferred_label'] = [r[0] for r in results]
    df['alt_labels'] = [r[1] for r in results]
    df['description_text'] = [r[2] for r in results]
    
    # Uri goc giu nguyen
    df = df.rename(columns={'id': 'uri', 'description': 'raw_description'})
    
    # Sap xep cot
    df = df[['uri', 'skill_id', 'preferred_label', 'alt_labels', 'description_text', 'raw_description']]
    
    # Luu file
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Da luu: {output_path}")
    print(f"  Columns: {list(df.columns)}")
    
    # In sample
    print("\nSample labels:")
    samples = [0, 7, 8, 53, 93, 151, 230, 288]  # Known skills from CSV
    for i in samples:
        if i < len(df):
            row = df.iloc[i]
            print(f"  [{i}] {row['preferred_label'][:50]}")
            print(f"       desc: {row['description_text'][:80]}...")
    
    return df


def verify_with_api(df, sample_size=10):
    """
    Verify heuristic labels bang ESCO API (lay 10 mau kiem tra).
    """
    import urllib.request
    
    print(f"\nVerify voi ESCO API ({sample_size} mau)...")
    
    # Chon random sample
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    correct = 0
    total = 0
    
    for _, row in sample.iterrows():
        uri = row['uri']
        heuristic_label = row['preferred_label']
        
        try:
            api_url = f"https://ec.europa.eu/esco/api/resource/skill?uri={uri}&language=en"
            with urllib.request.urlopen(api_url, timeout=10) as response:
                data = json.loads(response.read().decode())
                api_label = data.get('title', '')
                
                match = api_label.lower().strip() == heuristic_label.lower().strip()
                total += 1
                if match:
                    correct += 1
                
                status = "OK" if match else "MISMATCH"
                print(f"  [{status}] API: '{api_label}' | Heuristic: '{heuristic_label}'")
                
                time.sleep(0.5)  # Rate limit
        except Exception as e:
            print(f"  [ERROR] {uri}: {e}")
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.0f}%)")
    
    return correct, total


if __name__ == "__main__":
    input_file = DATA_DIR / "skills.csv"
    output_file = DATA_DIR / "skills_enriched.csv"
    
    df = enrich_esco_csv(input_file, output_file)
    
    # Optional: verify voi API
    if "--verify" in sys.argv:
        verify_with_api(df, sample_size=15)
    
    print("\nDone!")
