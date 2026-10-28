"""Run the production book work queues with controlled exceptions and stops.

--ref checks an older revision against the same regression. Only the tested
methods are extracted; the real thread pool and task lifetime management run.
Requires g++ on PATH. Each test process has a 20-second bound.
"""
import argparse
import hashlib
import pathlib
import shutil
import subprocess


def block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Git revision instead of the working tree")
    args = parser.parse_args()
    root = pathlib.Path(__file__).absolute().parents[3]
    variant = "ref-" + hashlib.sha256(args.ref.encode()).hexdigest()[:12] if args.ref else "current"
    output = root / "ignored" / "book-learning-regression" / variant
    output.mkdir(parents=True, exist_ok=True)

    def read(path):
        if args.ref:
            return subprocess.check_output(["git", "show", args.ref + ":" + path], cwd=root).decode("utf-8-sig")
        return (root / path).read_text(encoding="utf-8-sig")

    (output / "thread_pool.hpp").write_text(read("src/engine/thread_pool.hpp"), encoding="utf-8")
    book = read("src/engine/book.hpp")
    learning = read("src/engine/book_enlarge.hpp")
    (output / "book_parallel_method.hpp").write_text(
        "template<class F>\n" + block(book, "inline void parallel_for(int n,"), encoding="utf-8")
    (output / "book_learning_method.hpp").write_text(
        "template<class F>\n" + block(learning, "void book_learning_for_each(int n,"), encoding="utf-8")
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required on PATH")
    executable = output / "book_learning_tasks_regression.exe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", "-I" + str(output),
                    str(root / "src/tools/book_safety/book_learning_tasks_regression.cpp"),
                    "-o", str(executable)], check=True)
    raise SystemExit(subprocess.run([str(executable)], timeout=20).returncode)


if __name__ == "__main__":
    main()
