"""Test generator cancellation and invalid search results without random timing.

Compiles the production generator body with deterministic search/board doubles.
--ref reproduces pre-fix behavior; requires g++ on PATH.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

from check_selfplay_pause import block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Git revision instead of the working tree")
    args = parser.parse_args()
    root = Path(__file__).absolute().parents[3]
    path = "src/engine/random_board_generator.hpp"
    source = (subprocess.check_output(["git", "show", args.ref + ":" + path], cwd=root).decode("utf-8-sig")
              if args.ref else (root / path).read_text(encoding="utf-8-sig"))
    variant = hashlib.sha256(args.ref.encode()).hexdigest()[:12] if args.ref else "current"
    output = root / "ignored/issue653/generator" / variant
    output.mkdir(parents=True, exist_ok=True)
    (output / "random_board_generator_methods.hpp").write_text(
        block(source, "std::vector<int> random_board_generator("), encoding="utf-8")
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required on PATH")
    executable = output / "random_board_generator_regression.exe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(output),
                    str(Path(__file__).with_name("random_board_generator_regression.cpp")),
                    "-o", str(executable)], check=True)
    raise SystemExit(subprocess.run([str(executable)], timeout=10).returncode)


if __name__ == "__main__":
    main()
