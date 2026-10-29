"""Runner failure-path tests; no Egaroucid binary or GUI SDK needed."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

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

    def test_timeout_retains_output(self):
        with tempfile.TemporaryDirectory() as directory:
            class Args:
                timeout = 0.5
            ctx = Context(Args(), Path(directory))
            with self.assertRaises(CheckFailure):
                ctx.run([sys.executable, "-u", "-c", "import time; print('started'); time.sleep(10)"])
            self.assertIn("started", (Path(directory) / "process-01.stdout.log").read_text())


if __name__ == "__main__":
    unittest.main()
