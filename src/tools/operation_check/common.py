"""Small, dependency-free runner shared by the GUI and console checks."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback


ROOT = Path(__file__).absolute().parents[3]


def require(condition, message):
    # Do not use assert: python -O must still detect regressions.
    if not condition:
        raise CheckFailure(message)


class CheckFailure(RuntimeError):
    pass


@dataclass
class Case:
    name: str
    description: str
    check: object


class Context:
    def __init__(self, args, directory):
        self.args = args
        self.directory = directory
        self.process_index = 0

    def run(self, command, *, input="", cwd=None, timeout=None):
        """Keep evidence even when the child times out or crashes."""
        self.process_index += 1
        prefix = self.directory / f"process-{self.process_index:02d}"
        command = [str(item) for item in command]
        prefix.with_suffix(".command.json").write_text(
            json.dumps({"argv": command, "cwd": str(cwd or self.directory)}, indent=2),
            encoding="utf-8",
        )
        prefix.with_suffix(".stdin.txt").write_text(input, encoding="utf-8")
        with prefix.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, \
                prefix.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
            try:
                result = subprocess.run(
                    command, input=input, text=True, encoding="utf-8", errors="replace",
                    stdout=stdout, stderr=stderr, cwd=cwd or self.directory,
                    timeout=timeout if timeout is not None else self.args.timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise CheckFailure(f"Timed out after {error.timeout}s: {command[0]}") from error
        result.stdout = prefix.with_suffix(".stdout.log").read_text(encoding="utf-8", errors="replace")
        result.stderr = prefix.with_suffix(".stderr.log").read_text(encoding="utf-8", errors="replace")
        return result

    def application(self):
        """Stage only distributable files; never copy appdata or personal books."""
        application = self.directory.parent / "application"
        executable = application / self.args.exe.name
        if not executable.exists():
            application.mkdir()
            shutil.copytree(self.args.resources, application / "resources")
            for dll in self.args.exe.parent.glob("*.dll"):
                shutil.copy2(dll, application / dll.name)
            license_file = self.args.exe.parent / "LICENSE"
            if license_file.is_file():
                shutil.copy2(license_file, application / "LICENSE")
            shutil.copy2(self.args.exe, executable)
        return executable


def parser_for(suite, default_exe):
    parser = argparse.ArgumentParser(description=f"Egaroucid {suite} operation checks")
    parser.add_argument("--exe", type=Path, default=ROOT / "bin" / default_exe)
    parser.add_argument("--resources", type=Path,
                        help="Resources directory (default: resources next to --exe)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/operation_check" / suite)
    parser.add_argument("--case", action="append", default=[], metavar="NAME",
                        help="Run only this case; repeat to select several")
    parser.add_argument("--list", action="store_true", help="List cases without launching the app")
    parser.add_argument("--timeout", type=float, default=30, help="Child-process timeout in seconds")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run_suite(suite, parser, cases, argv=None):
    args = parser.parse_args(argv)
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate case name")
    unknown = set(args.case) - set(names)
    if unknown:
        parser.error("Unknown case: " + ", ".join(sorted(unknown)))
    selected = [case for case in cases if not args.case or case.name in args.case]
    if args.list:
        for case in selected:
            print(f"{case.name:24} {case.description}")
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.exe = args.exe.absolute()
    args.resources = (args.resources or args.exe.parent / "resources").absolute()
    if not args.exe.is_file():
        parser.error(f"Executable not found: {args.exe}")
    if not args.resources.is_dir():
        parser.error(f"Resources not found: {args.resources}")
    # A new directory for every run prevents stale evidence from passing a check.
    output = args.output_dir.absolute() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    report = {"suite": suite, "executable": str(args.exe), "resources": str(args.resources),
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    print(f"Results: {output}", flush=True)
    started = time.monotonic()
    for index, case in enumerate(selected, 1):
        elapsed = time.monotonic() - started
        eta = f"{elapsed / (index - 1) * (len(selected) - index + 1):.1f}s" if index > 1 else "estimating"
        print(f"[{index}/{len(selected)}] {case.name}: {case.description} (elapsed {elapsed:.1f}s, ETA {eta})", flush=True)
        directory = output / case.name
        directory.mkdir()
        case_started = time.monotonic()
        result = {"name": case.name, "status": "passed", "artifacts": str(directory)}
        try:
            case.check(Context(args, directory))
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result["seconds"] = round(time.monotonic() - case_started, 3)
        report["cases"].append(result)
        report["passed"] = all(item["status"] == "passed" for item in report["cases"])
        report["complete"] = len(report["cases"]) == len(selected)
        (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  {result['status'].upper()} ({result['seconds']:.2f}s)" +
              (f": {result['error']}" if "error" in result else ""), flush=True)
        if result["status"] == "failed" and args.fail_fast:
            break
    failed = sum(item["status"] == "failed" for item in report["cases"])
    print(f"{len(report['cases']) - failed} passed, {failed} failed; report: {output / 'results.json'}", flush=True)
    return 1 if failed else 0
