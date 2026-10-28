"""Test chthread against a built console executable, with and without pondering.

Run from the repository root:
    python src/tools/diagnostics/test_console_chthread.py --exe bin/Egaroucid_for_Console.exe

The executable needs its normal resources directory. Logs go to
build/console_chthread_tests by default. No opening book is loaded or changed.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).absolute().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "bin/Egaroucid_for_Console.exe")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/console_chthread_tests")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = [str(args.exe.absolute()), "-quiet", "-nobook", "-noise", "-thread", "2", "-hash", "18", "-l", "3"]
    commands = "\n".join([
        "help", "play f5", "chthread 4", "go", "undo", "chthread 1", "go", "undo",
        "chthread 3", "go", "undo", "transcript",
        "chthread 0", "chthread -1", "chthread abc", "chthread 2x", "chthread 2 3",
        "chthread 1.5", "chthread 999999999999999999999999", "chthread", "chthread 2", "exit", "",
    ])
    results = []
    for ponder in (False, True):
        label = "ponder" if ponder else "normal"
        result = subprocess.run(
            base + (["-ponder"] if ponder else []), input=commands, text=True,
            capture_output=True, encoding="utf-8", errors="replace", timeout=25, cwd=ROOT,
        )
        (args.output_dir / (label + ".stdout.log")).write_text(result.stdout, encoding="utf-8")
        (args.output_dir / (label + ".stderr.log")).write_text(result.stderr, encoding="utf-8")
        assert result.returncode == 0, (label, result.returncode, result.stderr[-1000:])
        assert "chthread <n_threads>" in result.stdout, (label, "help missing")
        counts = re.findall(r"thread size = (\d+)", result.stderr)
        assert counts == ["2", "4", "1", "3", "2"], (label, counts)
        assert result.stderr.count("[ERROR] thread count must be a positive integer") == 8, (label, "invalid input accepted")
        moves = re.findall(r"^([a-h][1-8])\s*$", result.stdout, re.MULTILINE)
        assert len(moves) == 3 and all(move in {"d6", "f4", "f6"} for move in moves), (label, moves)
        # undo preserves the redo history, which transcript prints in full.
        assert result.stdout.rstrip().splitlines()[-1] == "f5" + moves[-1], (label, "unexpected history")
        results.append({"mode": label, "passed": True, "thread_counts": counts, "search_moves": moves})
        print("PASS:", label, "thread changes, searches, invalid input, quiet output, and help")
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
