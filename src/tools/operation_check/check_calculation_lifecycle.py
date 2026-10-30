"""Exercise production GUI calculation methods with deterministic search futures.

Rendering and engine search are replaced by a small fixture. Main_scene methods
are extracted from the working tree or --ref, so old revisions reproduce bugs.
Requires g++ on PATH; output stays below ignored/issue653/lifecycle.
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
    parser.add_argument("--case", choices=("all", "cancel", "single", "resume", "random", "background"), default="all")
    args = parser.parse_args()
    root = Path(__file__).absolute().parents[3]
    path = "src/gui/main_scene.hpp"
    source = (subprocess.check_output(["git", "show", args.ref + ":" + path], cwd=root).decode("utf-8-sig")
              if args.ref else (root / path).read_text(encoding="utf-8-sig"))
    variant = hashlib.sha256(args.ref.encode()).hexdigest()[:12] if args.ref else "current"
    output = root / "ignored/issue653/lifecycle" / variant
    output.mkdir(parents=True, exist_ok=True)
    methods = [block(source, marker) for marker in (
        "void reset_ai()", "bool should_ai_move()", "void ai_move()",
        "void pause_calculating()", "void resume_calculating()", "void clear_interrupted_calculation_flags()",
        "void restart_interrupted_calculations()", "void reset_random_board_generator()",
        "void check_random_board_generater()", "void try_hint_get()", "void try_pv_get()",
        "void try_local_strategy_get()", "void try_local_strategy_policy_get()")]
    random_command = block(source, "if (getData().menu_elements.generate_random_board ||")
    command_body = random_command[random_command.index("{") + 1:-1]
    methods.append("void request_random_board() {" + command_body + "}")
    if "void start_random_board_generation()" in source:
        methods.append(block(source, "void start_random_board_generation()"))
    # The move fixture uses the same cancellation call as the production entry.
    assert "stop_calculating();" in block(source, "void move_processing(int_fast8_t cell)").splitlines()[1]
    (output / "calculation_lifecycle_methods.hpp").write_text("\n".join(methods), encoding="utf-8")
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required on PATH")
    executable = output / "calculation_lifecycle_regression.exe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(output),
                    str(Path(__file__).with_name("calculation_lifecycle_regression.cpp")),
                    "-o", str(executable)], check=True)
    raise SystemExit(subprocess.run([str(executable), args.case], timeout=10).returncode)


if __name__ == "__main__":
    main()
