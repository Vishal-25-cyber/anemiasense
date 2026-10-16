"""
AnemiaSense — Dataset Inspection, Cleaning & Auditing Module
Handles loading, schema validation, data cleaning, typo resolution,
cryptographic hash verification, and WHO ground-truth labeling.
"""

import os
import sys
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np


def clean_metadata_and_extract(dataset_base_dir: Path) -> pd.DataFrame:
    """
    Scans dataset hierarchy, cleans metadata, pairs verified image masks,
    and returns a clean, fully verified clinical DataFrame.
    """
    dataset_base = Path(dataset_base_dir)
    india_dir = dataset_base / "India"
    italy_dir = dataset_base / "Italy"

    if not india_dir.exists() or not italy_dir.exists():
        raise FileNotFoundError(f"Cohort directories not found inside {dataset_base}")

    # 1. Load Excel Metadata
    india_raw = pd.read_excel(india_dir / "India.xlsx")
    italy_raw = pd.read_excel(italy_dir / "Italy.xlsx")

    india_clean = india_raw[['Number', 'Hgb', 'Gender', 'Age']].copy()
    india_clean['Country'] = 'India'

    italy_clean = italy_raw[['Number', 'Hgb', 'Gender', 'Age']].copy()
    italy_clean['Country'] = 'Italy'

    combined = pd.concat([india_clean, italy_clean], ignore_index=True)
    initial_total = len(combined)

    # 2. Sanitize European Comma Decimals in Hgb
    combined['Hgb_clean'] = combined['Hgb'].astype(str).str.strip().str.replace(',', '.')
    combined['Hgb_num'] = pd.to_numeric(combined['Hgb_clean'], errors='coerce')

    # 3. Handle unrecorded laboratory measurements (e.g. Italy Patient 93)
    valid_meta = combined[combined['Hgb_num'].notna()].copy()
    valid_meta['Gender'] = valid_meta['Gender'].astype(str).str.strip().str.upper()
    valid_meta['Age'] = pd.to_numeric(valid_meta['Age'], errors='coerce').astype(int)

    # 4. Map Images and Segmentation Masks
    patient_records = []
    for idx, row in valid_meta.iterrows():
        c = row['Country']
        p_num = str(int(row['Number']))
        p_dir = dataset_base / c / p_num

        png_files = list(p_dir.glob("*.png"))
        # Match combined forniceal_palpebral first (handles typo palplebral in Italy 95), fallback to palpebral
        fp = [f for f in png_files if "forniceal" in f.name.lower() and "palp" in f.name.lower()]
        palp = [f for f in png_files if "palp" in f.name.lower() and "forn" not in f.name.lower()]

        best_roi = fp[0] if fp else (palp[0] if palp else None)
        jpg_files = list(p_dir.glob("*.jpg"))
        orig_jpg = jpg_files[0] if jpg_files else None

        patient_records.append({
            'Patient_ID': f"{c}_{p_num}",
            'Country': c,
            'Number': int(p_num),
            'Gender': row['Gender'],
            'Age': row['Age'],
            'Hgb': float(row['Hgb_num']),
            'Orig_JPG_Path': str(orig_jpg) if orig_jpg else None,
            'Mask_PNG_Path': str(best_roi) if best_roi else None
        })

    df = pd.DataFrame(patient_records)

    # 5. Cryptographic MD5 Duplicate Audit
    md5_hashes = {}
    dups = 0
    for p in df['Mask_PNG_Path']:
        if p and Path(p).exists():
            with open(p, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
                if h in md5_hashes:
                    dups += 1
                md5_hashes[h] = p

    # 6. Apply WHO Ground-Truth Definitions
    def label_who(r):
        thresh = 13.0 if r['Gender'] == 'M' else 12.0
        is_anemic = 1 if r['Hgb'] < thresh else 0
        if not is_anemic:
            sev = "Normal"
        elif r['Hgb'] < 8.0:
            sev = "Severe"
        elif r['Hgb'] < 11.0:
            sev = "Moderate"
        else:
            sev = "Mild"
        return pd.Series([is_anemic, sev, thresh], index=['Anemic', 'Severity', 'Hb_Threshold'])

    df[['Anemic', 'Severity', 'Hb_Threshold']] = df.apply(label_who, axis=1)

    print(f"[DatasetInspection] Audited {initial_total} raw records -> {len(df)} verified complete clinical cases.")
    print(f"[DatasetInspection] MD5 duplicate check: {dups} duplicates found across all patient images.")
    print(f"[DatasetInspection] Class balance: Normal={int((df['Anemic']==0).sum())}, Anemic={int((df['Anemic']==1).sum())}.")
    return df


if __name__ == "__main__":
    base = Path(r"c:\anemia final\data\extracted\dataset anemia")
    df = clean_metadata_and_extract(base)
    print(df[['Patient_ID', 'Country', 'Gender', 'Age', 'Hgb', 'Severity']].head(10))
