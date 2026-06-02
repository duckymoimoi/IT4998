"""
Fetch preferred labels tu ESCO API cho tat ca skills.
Dung concurrent requests de tang toc (~13,939 skills).

Output: data/skills_with_names.csv voi cac cot:
  - uri, skill_id, preferred_label, alt_labels, skill_type, description_en
"""

import pandas as pd
import json
import time
import sys
import os
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


API_BASE = "https://ec.europa.eu/esco/api/resource/skill"
MAX_WORKERS = 5  # Concurrent requests (nhe tay voi API)
BATCH_SAVE_EVERY = 500  # Luu tam moi 500 skills
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_FILE = DATA_DIR / "esco_api_cache.json"


def load_cache():
    """Load cache tu file"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Luu cache ra file"""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def fetch_skill_info(uri, max_retries=3):
    """
    Fetch thong tin skill tu ESCO API.
    
    Returns:
        dict voi keys: preferred_label, skill_type, description_en
    """
    for attempt in range(max_retries):
        try:
            url = f"{API_BASE}?uri={urllib.parse.quote(uri, safe='')}&language=en"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                
                preferred_label = data.get("title", "")
                
                # Alternative labels (EN)
                alt_labels_list = []
                alt_obj = data.get("alternativeLabel", {})
                if isinstance(alt_obj, dict) and "en" in alt_obj:
                    alt_labels_list = alt_obj["en"]
                    if isinstance(alt_labels_list, str):
                        alt_labels_list = [alt_labels_list]
                alt_labels = " | ".join(alt_labels_list) if alt_labels_list else ""
                
                # Skill type
                skill_type = ""
                type_links = data.get("_links", {}).get("hasSkillType", [])
                if type_links:
                    skill_type = type_links[0].get("title", "")
                
                # Description
                desc_en = ""
                desc_obj = data.get("description", {}).get("en", {})
                if desc_obj:
                    desc_en = desc_obj.get("literal", "")
                
                return {
                    "preferred_label": preferred_label,
                    "alt_labels": alt_labels,
                    "skill_type": skill_type,
                    "description_en": desc_en,
                }
                
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                wait = 2 ** attempt
                time.sleep(wait)
            elif e.code == 404:
                return {"preferred_label": "", "alt_labels": "", "skill_type": "", "description_en": ""}
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(1)
    
    return None


def fetch_all_skills(input_csv, output_csv):
    """
    Fetch preferred labels cho tat ca skills.
    """
    df = pd.read_csv(input_csv)
    print(f"Tong so skills: {len(df)}")
    
    # Load cache
    cache = load_cache()
    print(f"Cache hien co: {len(cache)} entries")
    
    # Filter skills chua co trong cache
    uris = df["id"].tolist()
    # Re-fetch skills missing alt_labels (old cache format)
    pending = [uri for uri in uris if uri not in cache or "alt_labels" not in cache.get(uri, {})]
    print(f"Can fetch: {len(pending)} skills (bao gom cache cu thieu alt_labels)")
    
    if not pending:
        print("Tat ca skills da co trong cache!")
    else:
        # Fetch song song
        completed = 0
        errors = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for uri in pending:
                future = executor.submit(fetch_skill_info, uri)
                futures[future] = uri
            
            for future in as_completed(futures):
                uri = futures[future]
                try:
                    result = future.result()
                    if result:
                        cache[uri] = result
                        completed += 1
                    else:
                        errors += 1
                except Exception as e:
                    errors += 1
                
                # Progress
                total_done = completed + errors
                if total_done % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = total_done / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - total_done) / rate if rate > 0 else 0
                    print(f"  Progress: {total_done}/{len(pending)} ({completed} OK, {errors} errors) "
                          f"Rate: {rate:.1f}/s ETA: {eta/60:.0f}min")
                
                # Save cache periodically
                if total_done % BATCH_SAVE_EVERY == 0:
                    save_cache(cache)
        
        # Final save
        save_cache(cache)
        elapsed = time.time() - start_time
        print(f"\nDone: {completed} fetched, {errors} errors in {elapsed:.0f}s")
    
    # Build output dataframe
    rows = []
    for _, row in df.iterrows():
        uri = row["id"]
        skill_id = uri.split("/")[-1] if "/" in uri else uri
        
        info = cache.get(uri, {})
        rows.append({
            "uri": uri,
            "skill_id": skill_id,
            "preferred_label": info.get("preferred_label", ""),
            "alt_labels": info.get("alt_labels", ""),
            "skill_type": info.get("skill_type", ""),
            "description_en": info.get("description_en", ""),
        })
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nDa luu: {output_csv}")
    print(f"Columns: {list(result_df.columns)}")
    print(f"Sample:")
    for i in [0, 7, 100, 500, 1000]:
        if i < len(result_df):
            r = result_df.iloc[i]
            alt_preview = r['alt_labels'][:50] + '...' if len(str(r['alt_labels'])) > 50 else r['alt_labels']
            print(f"  [{i}] {r['preferred_label'][:40]} | alts: {alt_preview}")
    
    return result_df


if __name__ == "__main__":
    input_file = DATA_DIR / "skills.csv"
    output_file = DATA_DIR / "skills_with_names.csv"
    
    # Cho phep chi fetch N skills dau tien (dev mode)
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])
        
        df = pd.read_csv(input_file)
        temp = DATA_DIR / "skills_limited.csv"
        df.head(limit).to_csv(temp, index=False)
        input_file = temp
        print(f"Dev mode: chi fetch {limit} skills dau tien")
    
    fetch_all_skills(input_file, output_file)
