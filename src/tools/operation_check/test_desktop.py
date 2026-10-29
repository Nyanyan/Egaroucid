import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from common import CheckFailure
from desktop import report, validate_case


class DesktopCoverageTests(unittest.TestCase):
    def test_internal_smoke_result_cannot_pass_desktop_case(self):
        with self.assertRaises(CheckFailure):
            validate_case({"required": ["clicked"]}, {"status": "passed", "actions": [],
                          "assertions": [{"name": "clicked", "passed": True}]})

    def test_missing_postcondition_cannot_pass(self):
        with self.assertRaises(CheckFailure):
            validate_case({"required": ["clicked", "saved"]}, {"status": "passed",
                          "actions": [{"driver": "os-input", "completed": True}],
                          "assertions": [{"name": "clicked", "passed": True, "expected": {"scene": "Main_scene"}}]})

    def test_completed_action_and_postconditions_pass(self):
        validate_case({"required": ["clicked"]}, {"status": "passed",
                      "actions": [{"driver": "os-input", "completed": True}],
                      "assertions": [{"name": "clicked", "passed": True, "expected": {"discs": 5}}]})

    def test_not_run_required_case_fails_release_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "run.json").write_text(json.dumps({"token": "current", "cases": {
                "installer-new": {"description": "installer", "required": ["installed"]}}}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(report(path), 1)
            result = json.loads((path / "release-results.json").read_text(encoding="utf-8"))
            self.assertFalse(result["passed"])
            self.assertEqual(result["cases"][0]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
