"""Compile and run the Windows shortcut state regression tests without Siv3D.

Run from the repository root with Python and MinGW g++ on PATH:
    python src/tools/diagnostics/test_shortcut_key_state.py

The observer and matcher are extracted from the production header. Only the
Siv3D string/name adapter and Win32 input queries are replaced by test doubles.
--source can select an older header to verify that the regression is detected.
"""

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).absolute().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "src/gui/function/shortcut_key.hpp")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/shortcut_key_state_tests")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8-sig")
    sections = [
        ("inline bool windows_shortcut_modifier_vk(", "inline bool windows_shortcut_vk_to_key_name("),
        ("inline bool windows_vk_pressed(", "inline LRESULT CALLBACK windows_shortcut_subclass_proc("),
        ("inline bool windows_shortcut_vk_pressed(", "inline std::string windows_shortcut_vk_label("),
        ("inline bool windows_shortcut_key_name_found(", "inline std::vector<String> get_windows_shortcut_inputs_for_diagnostics("),
    ]
    implementation = "\n".join(source[source.index(start):source.index(end)] for start, end in sections)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    (args.build_dir / "shortcut_key_state_under_test.hpp").write_text(implementation, encoding="utf-8")
    executable = args.build_dir / "test_shortcut_key_state.exe"
    subprocess.run([
        args.compiler, "-std=c++20", "-O1", "-Wall", "-Wextra",
        "-I", str(args.build_dir),
        str(Path(__file__).with_suffix(".cpp")), "-o", str(executable), "-luser32",
    ], check=True, cwd=ROOT)
    return subprocess.run([str(executable)], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
