"""Runner failure-path tests; no Egaroucid binary or GUI SDK needed."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from common import Case, Context, CheckFailure, parser_for, require, run_suite


class RunnerTests(unittest.TestCase):
    def test_selection_failure_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            calls = []
            def fail(ctx):
                require(False, "deliberate regression")
            cases = [Case("bad", "failure", fail), Case("good", "success", lambda ctx: calls.append(1))]
            argv = ["--exe", sys.executable, "--resources", directory, "--output-dir", directory]
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_suite("test", parser_for("test", "unused"), cases, argv)
            self.assertEqual(code, 1)
            self.assertEqual(calls, [1])
            report = json.loads(next(path.glob("*/results.json")).read_text())
            self.assertEqual([r["status"] for r in report["cases"]], ["failed", "passed"])
            self.assertTrue(report["complete"])
            self.assertIn("deliberate regression", report["cases"][0]["error"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_suite("test", parser_for("test", "unused"), cases,
                                           argv + ["--case", "good"]), 0)

    def test_unknown_case_is_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            run_suite("test", parser_for("test", "unused"), [], ["--case", "typo"])
        self.assertEqual(error.exception.code, 2)

    def test_fail_fast_marks_report_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            def fail(ctx):
                require(False, "regression")
            cases = [Case("bad", "fails", fail), Case("later", "must not run", lambda ctx: self.fail())]
            argv = ["--exe", sys.executable, "--resources", directory, "--output-dir", directory, "--fail-fast"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_suite("test", parser_for("test", "unused"), cases, argv), 1)
            report = json.loads(next(Path(directory).glob("*/results.json")).read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report["passed"])

    def test_invalid_timeout(self):
        for timeout in ("0", "-1", "nan", "inf"):
            with self.subTest(timeout=timeout), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                run_suite("test", parser_for("test", "unused"), [], ["--timeout", timeout])
            self.assertEqual(error.exception.code, 2)

    def test_timeout_retains_output(self):
        with tempfile.TemporaryDirectory() as directory:
            class Args:
                timeout = 0.5
            ctx = Context(Args(), Path(directory))
            # Exercise log retention without assuming the interpreter can start
            # and print within the process timeout on a busy machine.
            def output_then_timeout(command, **kwargs):
                kwargs["stdout"].write("started\n")
                kwargs["stderr"].write("diagnostic\n")
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            with patch("common.subprocess.run", side_effect=output_then_timeout), self.assertRaises(CheckFailure):
                ctx.run(["test-child"])
            self.assertIn("started", (Path(directory) / "process-01.stdout.log").read_text())
            self.assertIn("diagnostic", (Path(directory) / "process-01.stderr.log").read_text())

    def test_timeout_rejects_hung_process(self):
        with tempfile.TemporaryDirectory() as directory:
            class Args:
                timeout = 0.5
            ctx = Context(Args(), Path(directory))
            with self.assertRaises(CheckFailure) as error:
                ctx.run([sys.executable, "-c", "import time; time.sleep(10)"])
            self.assertIsInstance(error.exception.__cause__, subprocess.TimeoutExpired)


if __name__ == "__main__":
    unittest.main()
