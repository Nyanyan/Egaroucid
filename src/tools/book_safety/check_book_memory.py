"""Check adaptive book-table rebuilds without exhausting the host's RAM.

The C++ regression injects memory snapshots and malloc failures into actual
table rebuilds; it also checks the platform query and concurrent reservations.
Requires g++ on PATH. The test run has a 45-second bound.
"""
import pathlib
import shutil
import subprocess


def main():
    directory = pathlib.Path(__file__).absolute().parent
    root = directory.parents[2]
    output = root / "ignored" / "book-memory-regression"
    output.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required on PATH")
    executable = output / "book_memory_regression.exe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-march=native", "-pthread", "-I" + str(root / "src"),
                    str(directory / "book_memory_regression.cpp"), "-o", str(executable)], check=True)
    raise SystemExit(subprocess.run([str(executable)], timeout=45).returncode)


if __name__ == "__main__":
    main()
