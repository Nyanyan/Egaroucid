"""Run the real book-reference transition methods with deterministic save failures.

Use --ref <commit> to check an older implementation against the same regression.
Requires g++ on PATH; outputs are written below ignored/pr643-fixes/reference.
"""
import argparse
import pathlib
import shutil
import subprocess


def block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Git revision instead of the working tree")
    args = parser.parse_args()
    root = pathlib.Path(__file__).absolute().parents[3]
    path = "src/gui/book_scene.hpp"
    if args.ref:
        source = subprocess.check_output(["git", "show", args.ref + ":" + path], cwd=root).decode("utf-8-sig")
    else:
        source = (root / path).read_text(encoding="utf-8-sig")
    source = source[source.index("class Refer_book :"):]
    if "void start_reference_change()" in source:
        methods = "\n".join(block(source, "void " + name + "()") for name in
                            ["start_book_import", "start_reference_change", "finish_book_save"])
    else:
        # Run the pre-fix click handler unchanged; the same retention assertions must fail.
        action = block(source, "if (formatted_book &&")
        methods = "void start_reference_change() " + action[action.index("{"):] + "\nvoid finish_book_save() {}\n"
    output = root / "ignored" / "pr643-fixes" / "reference"
    output.mkdir(parents=True, exist_ok=True)
    (output / "book_reference_methods.hpp").write_text(methods, encoding="utf-8")
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required on PATH")
    exe = output / "book_reference_regression.exe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", "-I" + str(output),
                    str(pathlib.Path(__file__).absolute().with_name("book_reference_regression.cpp")), "-o", str(exe)], check=True)
    raise SystemExit(subprocess.run([str(exe)], timeout=15).returncode)


if __name__ == "__main__":
    main()
