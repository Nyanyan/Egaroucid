"""Check ordered book-learning candidates on openings with shared descendants.

The production engine runs without evaluation files or AI searches. --ref builds
an older revision against the same fixture and order/metadata assertions.
Requires g++ on PATH. The test process has a 30-second bound.
"""
import argparse
import hashlib
import io
import pathlib
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Git revision instead of the working tree")
    args = parser.parse_args()
    root = pathlib.Path(__file__).absolute().parents[3]
    variant = "ref-" + hashlib.sha256(args.ref.encode()).hexdigest()[:12] if args.ref else "current"
    output = root / "ignored" / "book-scan-regression" / variant
    output.mkdir(parents=True, exist_ok=True)
    include = root / "src"
    if args.ref:
        archive = subprocess.check_output(["git", "archive", "--format=zip", args.ref, "src/engine"], cwd=root)
        with zipfile.ZipFile(io.BytesIO(archive)) as files:
            files.extractall(output)
        include = output / "src"
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required on PATH")
    executable = output / "book_scan_regression.exe"
    flags = []
    if sys.platform == "win32":
        # MinGW 13 can emit aligned AVX argument copies on Win64's 16-byte stack.
        # Run this data-free fixture with the Generic engine; MSVC checks SIMD.
        flags = ["-DHAS_NO_AVX2", "-mno-avx", "-mno-avx2", "-mno-avx512f"]
    subprocess.run([compiler, "-std=c++20", "-O2", "-march=native", "-pthread", *flags, "-I" + str(include),
                    str(root / "src/tools/book_safety/book_scan_regression.cpp"),
                    "-o", str(executable)], check=True)
    raise SystemExit(subprocess.run([str(executable)], timeout=30).returncode)


if __name__ == "__main__":
    main()
