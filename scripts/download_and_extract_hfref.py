"""
Script to download and extract the HFrEF 5-fold dataset from Zenodo.
Official URL: https://zenodo.org/records/16934966
File: HFrEF_5_folds.zip (~278 MB)
Destination: data/hfref/
"""
import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

ZENODO_ZIP_URL = "https://zenodo.org/api/records/16934966/files/HFrEF_5_folds.zip/content"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "hfref"
ZIP_PATH = TARGET_DIR / "HFrEF_5_folds.zip"
EXTRACT_DIR = TARGET_DIR / "extracted"

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 280000000:
        print(f"[INFO] File already downloaded: {dest_path} ({dest_path.stat().st_size} bytes)")
        return

    print(f"[INFO] Downloading HFrEF dataset from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    start_time = time.time()
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1MB chunks
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                mb = downloaded / (1024 * 1024)
                tot_mb = total_size / (1024 * 1024)
                print(f"\rDownloading: {mb:.1f}/{tot_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\rDownloaded: {mb:.1f} MB", end="", flush=True)
                
    elapsed = time.time() - start_time
    print(f"\n[INFO] Download finished in {elapsed:.1f}s: {dest_path} ({dest_path.stat().st_size} bytes)")

def extract_zip(zip_path: Path, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zip_path} into {extract_dir}...")
    start_time = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    elapsed = time.time() - start_time
    print(f"[INFO] Extraction complete in {elapsed:.1f}s.")

if __name__ == "__main__":
    download_file(ZENODO_ZIP_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, EXTRACT_DIR)
