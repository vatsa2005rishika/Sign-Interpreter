"""
get_vendor.py - run ONCE while the laptop has internet.

Downloads the web-page libraries (React, Babel, Tailwind, framer-motion, lucide) into
frontend/vendor/ so the web page works on a hotspot with NO internet.
(These are the same libraries your page already used, just saved locally.
 React is the smaller "production" build.)

Run:  python get_vendor.py
"""
import sys
import urllib.request
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "frontend" / "vendor"
FILES = {
    "tailwind.js":               "https://cdn.tailwindcss.com",
    "react.production.min.js":   "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom.production.min.js": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
    "babel.min.js":              "https://unpkg.com/@babel/standalone/babel.min.js",
    "framer-motion.js":          "https://unpkg.com/framer-motion@10.16.4/dist/framer-motion.js",
    "lucide.js":                 "https://unpkg.com/lucide@latest",
}

VENDOR.mkdir(parents=True, exist_ok=True)
bad = 0
for name, url in FILES.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        if len(data) < 2000 or data.lstrip()[:1] == b"<":
            raise ValueError(f"unexpected content ({len(data)} bytes)")
        (VENDOR / name).write_bytes(data)
        print(f"ok   {name:30s} {len(data)//1024} KB")
    except Exception as e:
        bad += 1
        print(f"FAIL {name:30s} {e}")
print("\nAll done." if not bad else f"\n{bad} file(s) failed - check your internet and run again.")
sys.exit(1 if bad else 0)
