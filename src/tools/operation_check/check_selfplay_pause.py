"""Check the real AI-turn and graph-input conditions when calculation is paused.

--ref runs the pre-fix update expression against the same regression. The small
fixture replaces rendering and search, not the conditions taken from Main_scene.
Requires g++ on PATH. Outputs go below ignored/issue653/regression.
"""
import argparse
import hashlib
from pathlib import Path
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
    root = Path(__file__).absolute().parents[3]
    path = "src/gui/main_scene.hpp"
    source = (subprocess.check_output(["git", "show", args.ref + ":" + path], cwd=root).decode("utf-8-sig")
              if args.ref else (root / path).read_text(encoding="utf-8-sig"))
    variant = hashlib.sha256(args.ref.encode()).hexdigest()[:12] if args.ref else "current"
    output = root / "ignored/issue653/regression" / variant
    output.mkdir(parents=True, exist_ok=True)
    if "bool should_ai_move()" in source:
        predicate = block(source, "bool should_ai_move()")
    else:
        start = source.index("        bool ai_should_move =")
        end = source.index("        bool ignore_move =", start)
        predicate = "bool should_ai_move() {\n" + source[start:end] + "return ai_should_move;\n}"
    ignore_start = source.index("        bool ignore_move =")
    ignore_end = source.index(";", ignore_start) + 1
    graph_start = source.index("        bool graph_interact_ignore =")
    graph_end = source.index("        update_n_discs();", graph_start)
    graph = ("void route_graph_input() {\nconst bool ai_should_move = should_ai_move();\n"
             + source[ignore_start:ignore_end] + "\n" + source[graph_start:graph_end] + "\n}")
    (output / "selfplay_pause_methods.hpp").write_text(predicate + "\n" + graph, encoding="utf-8")
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required on PATH")
    executable = output / "selfplay_pause_regression.exe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(output),
                    str(Path(__file__).with_name("selfplay_pause_regression.cpp")),
                    "-o", str(executable)], check=True)
    raise SystemExit(subprocess.run([str(executable)], timeout=10).returncode)


if __name__ == "__main__":
    main()
