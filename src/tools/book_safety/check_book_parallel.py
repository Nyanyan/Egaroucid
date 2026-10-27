"""Check concurrent accesses and results of the real book traversal code.

The test wrapper observes access intervals inside the actual table locks; it is
not ThreadSanitizer. --ref checks an older book.hpp against the same regression.
Requires g++ on PATH. Each test run has a 45-second bound.
"""
import argparse
import pathlib
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Git revision for book.hpp instead of the working tree")
    args = parser.parse_args()
    directory = pathlib.Path(__file__).absolute().parent
    root = directory.parents[2]
    output = root / "ignored" / "pr643-fixes" / "parallel"
    output.mkdir(parents=True, exist_ok=True)
    include = root / "src"
    if args.ref:
        include = output / "baseline"
        shutil.copytree(root / "src" / "engine", include / "engine", dirs_exist_ok=True)
        original = subprocess.check_output(["git", "show", args.ref + ":src/engine/book.hpp"], cwd=root)
        (include / "engine" / "book.hpp").write_bytes(original)
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required on PATH")
    executable = output / "book_parallel_regression.exe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-march=native", "-pthread", "-I" + str(include),
                    str(directory / "book_parallel_regression.cpp"), "-o", str(executable)], check=True)
    raise SystemExit(subprocess.run([str(executable)], timeout=45).returncode)


if __name__ == "__main__":
    main()
