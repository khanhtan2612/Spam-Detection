"""Kiểm tra chuẩn bị dependency bằng đúng Python của kernel, không cài thật."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.notebook_setup import ensure_dependencies


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NotebookSetupTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_available_dependencies_do_not_trigger_installation(self):
        with patch("scripts.notebook_setup.importlib.util.find_spec", return_value=object()), \
             patch("scripts.notebook_setup.subprocess.check_call") as install:
            ensure_dependencies(PROJECT_ROOT)
        install.assert_not_called()

    def test_missing_dependency_is_installed_into_active_interpreter(self):
        installed = False

        def find_spec(name):
            return None if name == "joblib" and not installed else object()

        def complete_install(*args, **kwargs):
            nonlocal installed
            installed = True

        with patch("scripts.notebook_setup.importlib.util.find_spec", side_effect=find_spec), \
             patch("scripts.notebook_setup.subprocess.check_call", side_effect=complete_install) as install:
            ensure_dependencies(PROJECT_ROOT)

        install.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
            cwd=PROJECT_ROOT,
        )

    def test_failed_installation_is_reported_with_active_interpreter(self):
        failure = subprocess.CalledProcessError(1, [sys.executable, "-m", "pip"])
        with patch("scripts.notebook_setup.importlib.util.find_spec", return_value=None), \
             patch("scripts.notebook_setup.subprocess.check_call", side_effect=failure):
            with self.assertRaises(RuntimeError) as raised:
                ensure_dependencies(PROJECT_ROOT)

        self.assertIn(sys.executable, str(raised.exception))

    def test_dependencies_still_missing_after_installation_are_reported(self):
        with patch("scripts.notebook_setup.importlib.util.find_spec", return_value=None), \
             patch("scripts.notebook_setup.subprocess.check_call") as install:
            with self.assertRaises(RuntimeError) as raised:
                ensure_dependencies(PROJECT_ROOT)

        install.assert_called_once()
        self.assertIn("joblib", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
