"""A missing, stale, or failed GUI response must never become a green result."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import CheckFailure
from gui import native, require_probe_support


class FakeGUI:
    process_index = 0

    def __init__(self, directory, change):
        self.directory = directory
        self.change = change

    def application(self):
        return Path("unused.exe")

    def run(self, command, cwd):
        self.process_index += 1
        request = json.loads((self.directory / "request.json").read_text())
        response = {"protocol": 1, "token": request["token"], "case": request["case"],
                    "passed": True, "portable": True}
        response.update(self.change)
        (self.directory / "native-result.json").write_text(json.dumps(response))
        return SimpleNamespace(returncode=0)


class GUIResultTests(unittest.TestCase):
    def test_only_matching_success_is_accepted(self):
        for change in ({}, {"token": "stale"}, {"case": "other"}, {"protocol": 99},
                       {"passed": False, "error": "injected regression"}, {"portable": False}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory, \
                    patch("gui.require_probe_support"):
                path = Path(directory)
                (path / "appdata").mkdir()
                (path / "appdata/shortcut_key.json").write_text("{}")
                ctx = FakeGUI(path, change)
                if change:
                    with self.assertRaises(CheckFailure):
                        native(ctx, "startup")
                else:
                    self.assertTrue(native(ctx, "startup")["passed"])

    def test_missing_shutdown_evidence_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch("gui.require_probe_support"), self.assertRaises(CheckFailure):
            native(FakeGUI(Path(directory), {}), "startup")

    def test_missing_native_result_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch("gui.require_probe_support"), self.assertRaises(CheckFailure):
            ctx = FakeGUI(Path(directory), {})
            ctx.run = lambda *args, **kwargs: SimpleNamespace(returncode=0)
            native(ctx, "startup")

    def test_old_binary_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "old.exe"
            executable.write_bytes(b"MZ old GUI without a probe")
            with self.assertRaises(CheckFailure):
                require_probe_support(executable)
            ctx = FakeGUI(Path(directory), {})
            ctx.application = lambda: executable
            ctx.run = lambda *args, **kwargs: self.fail("An unsupported binary was launched")
            with self.assertRaises(CheckFailure):
                native(ctx, "startup")
            executable.write_bytes("egaroucid-operation-check-v1".encode("utf-32-le"))
            require_probe_support(executable)


if __name__ == "__main__":
    unittest.main()
