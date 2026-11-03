"""
High-Performance Resilient Multi-Threaded Downloader and Extractor
for CirCor DigiScope Phonocardiogram Dataset (v1.0.3) from PhysioNet.
"""
import os
import sys
import time
import json
import zipfile
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import concurrent.futures
import shutil
import threading

DATASET_URL = "https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip"
ZIP_PATH = os.path.join("data", "raw", "zips", "the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
STATE_PATH = os.path.join("data", "raw", "zips", "download_state.json")
EXTRACT_TARGET = os.path.join("data", "raw", "circor")
BLOCK_SIZE = 2 * 1024 * 1024  # 2 MB
NUM_WORKERS = 48

file_lock = threading.Lock()
state_lock = threading.Lock()

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=2, pool_maxsize=2)
        s.mount('https://', adapter)
        s.mount('http://', adapter)
        thread_local.session = s
    return thread_local.session

def get_remote_size(url):
    s = requests.Session()
    r = s.head(url, allow_redirects=True, timeout=20)
    return int(r.headers.get('Content-Length', 0))

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'r') as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_state(completed_set):
    with state_lock:
        with open(STATE_PATH, 'w') as f:
            json.dump(list(completed_set), f)

def download_block(url, block_id, start_byte, end_byte, output_path, max_retries=10):
    session = get_session()
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'Range': f'bytes={start_byte}-{end_byte}'
            }
            r = session.get(url, headers=headers, timeout=90)
            if r.status_code in (200, 206):
                data = r.content
                expected = end_byte - start_byte + 1
                if len(data) != expected:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                
                with file_lock:
                    with open(output_path, 'r+b') as out_f:
                        out_f.seek(start_byte)
                        out_f.write(data)
                        out_f.flush()
                return block_id, len(data)
        except Exception as e:
            time.sleep(0.3 * (attempt + 1))
            if attempt == max_retries - 1:
                raise RuntimeError(f"Block {block_id} failed after {max_retries} retries: {e}")

def download_dataset():
    total_size = get_remote_size(DATASET_URL)
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    
    print("=" * 60, flush=True)
    print(" CirCor DigiScope Dataset Downloader", flush=True)
    print(f" Total size: {total_size / (1024*1024):.2f} MB ({total_size:,} bytes)", flush=True)
    print(f" Workers: {NUM_WORKERS} | Block size: {BLOCK_SIZE / (1024*1024):.1f} MB", flush=True)
    print("=" * 60, flush=True)
    
    if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) != total_size:
        with open(ZIP_PATH, 'wb') as f:
            f.seek(total_size - 1)
            f.write(b'\0')
        if os.path.exists(STATE_PATH):
            os.remove(STATE_PATH)
            
    num_blocks = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    completed_blocks = load_state()
    
    blocks_to_download = []
    for b_idx in range(num_blocks):
        if b_idx in completed_blocks:
            continue
        start = b_idx * BLOCK_SIZE
        end = min((b_idx + 1) * BLOCK_SIZE - 1, total_size - 1)
        blocks_to_download.append((b_idx, start, end))
        
    print(f"[1/4] Blocks already cached: {len(completed_blocks)}/{num_blocks}. Downloading remaining {len(blocks_to_download)} blocks...", flush=True)
    
    if blocks_to_download:
        start_time = time.time()
        last_report = start_time
        downloaded_since_start = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_map = {
                executor.submit(download_block, DATASET_URL, b_idx, start, end, ZIP_PATH): b_idx
                for b_idx, start, end in blocks_to_download
            }
            
            for future in concurrent.futures.as_completed(future_map):
                b_idx, nbytes = future.result()
                completed_blocks.add(b_idx)
                downloaded_since_start += nbytes
                save_state(completed_blocks)
                
                now = time.time()
                if now - last_report >= 1.5 or len(completed_blocks) == num_blocks:
                    elapsed = now - start_time
                    pct = (len(completed_blocks) / num_blocks) * 100
                    speed = (downloaded_since_start) / (1024 * 1024 * max(0.1, elapsed))
                    print(f"      Blocks: {len(completed_blocks)}/{num_blocks} ({pct:.1f}%) - Speed: {speed:.2f} MB/s", flush=True)
                    last_report = now

        elapsed = time.time() - start_time
        print(f"      Download completed in {elapsed:.1f}s", flush=True)
        
    if os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)

def verify_and_extract(zip_path, extract_dir):
    print(f"\n[2/4] Verifying zip archive integrity: {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise ValueError(f"Corrupt file found in zip archive: {bad_file}")
        print("      Zip archive integrity check passed.", flush=True)
        
        print(f"\n[3/4] Extracting files into target directory: {extract_dir}", flush=True)
        os.makedirs(extract_dir, exist_ok=True)
        zf.extractall(extract_dir)
        print("      Extraction finished.", flush=True)

def reorganize_if_nested(extract_dir):
    print(f"\n[4/4] Standardizing directory layout...", flush=True)
    items = os.listdir(extract_dir)
    if len(items) == 1 and os.path.isdir(os.path.join(extract_dir, items[0])):
        nested_dir = os.path.join(extract_dir, items[0])
        print(f"      Found nested directory: {items[0]}, moving contents up to {extract_dir}...", flush=True)
        for sub_item in os.listdir(nested_dir):
            src = os.path.join(nested_dir, sub_item)
            dst = os.path.join(extract_dir, sub_item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(nested_dir)
        print("      Directory structure flattened successfully.", flush=True)
    else:
        print(f"      Layout already formatted: {items[:5]}", flush=True)

def main():
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) == 471283852 and not os.path.exists(STATE_PATH):
        try:
            print("Checking existing zip file integrity...", flush=True)
            with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
                if zf.testzip() is None:
                    print("Zip file is valid.", flush=True)
                else:
                    raise Exception("Zip test failed.")
        except Exception:
            download_dataset()
    else:
        download_dataset()
        
    verify_and_extract(ZIP_PATH, EXTRACT_TARGET)
    reorganize_if_nested(EXTRACT_TARGET)
    print("\nDataset ready in:", os.path.abspath(EXTRACT_TARGET), flush=True)

if __name__ == "__main__":
    main()
