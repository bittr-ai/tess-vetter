from __future__ import annotations

import subprocess
import sys


def test_python_module_entrypoint_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tess_vetter", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout

