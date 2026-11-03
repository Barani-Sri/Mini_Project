"""
Launcher script to start the LubDub Murmur web server.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print(" Starting LubDub Murmur Web Server...")
    print(" URL: http://127.0.0.1:8000 (and http://localhost:8000)")
    print("=" * 70)
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=False)

