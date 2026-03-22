"""
Builds submission.zip from submission/ folder.

Steps:
  1. Verifies run.py has no banned imports
  2. Checks best.pt exists
  3. Creates submission.zip with run.py and best.pt at root level
  4. Verifies zip structure

Usage:
    python make_submission.py
    python make_submission.py --weights runs/detect/grocery/weights/best.pt
"""

import json
import zipfile
import argparse
from pathlib import Path

BANNED_IMPORTS = [
    "import os", "import sys", "import subprocess", "import socket",
    "import ctypes", "import builtins", "import importlib",
    "import pickle", "import marshal", "import shelve", "import shutil",
    "import yaml", "import requests", "import urllib", "import http",
    "import multiprocessing", "import threading", "import signal", "import gc",
    "import code", "import codeop", "import pty",
    "from os ", "from sys ", "from subprocess", "from socket",
    "from pickle", "from shutil", "from yaml", "from requests",
    "from urllib", "from multiprocessing", "from threading",
]
BANNED_CALLS = ["eval(", "exec(", "compile(", "__import__("]


def scan_py_file(path: Path) -> list[str]:
    violations = []
    text = path.read_text()
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for banned in BANNED_IMPORTS + BANNED_CALLS:
            if banned in line:
                violations.append(f"  Line {line_no}: {line.rstrip()}")
                break
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["submission/model_l.onnx", "submission/model_refs.onnx"],
                        help="ONNX model files to include")
    parser.add_argument("--out", default="submission.zip")
    args = parser.parse_args()

    sub_dir = Path("submission")
    run_py = sub_dir / "run.py"
    out_zip = Path(args.out)

    onnx_files = [Path(m) for m in args.models]

    # --- Sanity checks ---
    assert run_py.exists(), f"Missing {run_py}"
    assert onnx_files, f"No .onnx files found in {sub_dir}"

    print("Scanning run.py for banned imports/calls ...")
    violations = scan_py_file(run_py)
    if violations:
        print("FAIL — banned patterns found in run.py:")
        for v in violations:
            print(v)
        raise SystemExit(1)
    print("  run.py: CLEAN")

    total_uncompressed = sum(f.stat().st_size for f in onnx_files)
    total_mb = total_uncompressed / 1024 / 1024
    print(f"  ONNX files: {[f.name for f in onnx_files]}")
    print(f"  Total uncompressed: {total_mb:.1f} MB")
    if total_mb > 420:
        print(f"WARNING: exceeds 420 MB uncompressed limit ({total_mb:.1f} MB)")
        raise SystemExit(1)

    # --- Build zip ---
    print(f"\nBuilding {out_zip} ...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(run_py, "run.py")
        for f in onnx_files:
            zf.write(f, f.name)

    # --- Verify structure ---
    print("\nZip contents:")
    uncompressed_total = 0
    with zipfile.ZipFile(out_zip) as zf:
        for info in zf.infolist():
            print(f"  {info.filename}  ({info.file_size / 1024 / 1024:.1f} MB)")
            uncompressed_total += info.file_size

    compressed_mb = out_zip.stat().st_size / 1024 / 1024
    print(f"\nUncompressed: {uncompressed_total / 1024 / 1024:.1f} MB | Compressed: {compressed_mb:.1f} MB")
    print(f"Submission ready: {out_zip.resolve()}")


if __name__ == "__main__":
    main()
