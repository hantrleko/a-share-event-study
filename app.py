"""Railway starts this file when no separate start command is configured."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    port = env.get("PORT", "8501")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "app" / "streamlit_app.py"),
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
