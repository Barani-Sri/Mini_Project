import re
import sys
from pathlib import Path

js_path = Path("src/web/static/app.js")
html_path = Path("src/web/static/index.html")

js_content = js_path.read_text(encoding="utf-8")
html_content = html_path.read_text(encoding="utf-8")

# Extract section from 'Integrated Result Controller' onwards
start_idx = js_content.find("Integrated Result Controller")
int_js = js_content[start_idx:]

ids = set(re.findall(r"document\.getElementById\(['\"]([^'\"]+)['\"]\)", int_js))

print(f"Checking {len(ids)} element IDs from app.js in index.html:")
missing = []
for el_id in sorted(ids):
    if f'id="{el_id}"' in html_content or f"id='{el_id}'" in html_content:
        print(f"  [OK] id=\"{el_id}\"")
    else:
        missing.append(el_id)
        print(f"  [MISSING] id=\"{el_id}\"")

if missing:
    print(f"\nERROR: {len(missing)} missing IDs: {missing}")
    sys.exit(1)
else:
    print(f"\nSUCCESS: All {len(ids)} DOM IDs exist in index.html!")
