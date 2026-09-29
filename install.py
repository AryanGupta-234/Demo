#!/usr/bin/env python3
"""
One-command installer for the Pre-CAB Validator.

Usage:
    python install.py

The installer creates an isolated .venv in the repository and installs the
complete project dependency set (all runtime/document/API/embedding packages
plus development/test tooling).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n> " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main() -> int:
    print("=" * 72)
    print("Pre-CAB Validator — One-Command Installer")
    print("=" * 72)
    print(f"Platform : {platform.system()} {platform.machine()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Project  : {ROOT}")

    if sys.version_info < (3, 11):
        print("\nERROR: Python 3.11 or newer is required.")
        print("Install Python 3.11+ and run this installer again.")
        return 1

    if not (ROOT / "pyproject.toml").exists():
        print("\nERROR: pyproject.toml was not found.")
        print("Run this installer from the repository root.")
        return 1

    if not VENV.exists():
        print("\n[1/4] Creating isolated virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV)])
    else:
        print("\n[1/4] Existing .venv found — reusing it.")

    py = venv_python()
    if not py.exists():
        print(f"ERROR: Virtual-environment Python was not found: {py}")
        return 1

    print("\n[2/4] Upgrading packaging tools...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    print("\n[3/4] Installing the complete dependency set...")
    run([str(py), "-m", "pip", "install", "-e", ".[all,dev]"])

    print("\n[4/4] Verifying the installation...")
    run([str(py), "-m", "compileall", "-q", "src", "scripts", "main.py"])
    run([str(py), "-m", "pytest"])

    print("\n" + "=" * 72)
    print("INSTALLATION COMPLETE")
    print("=" * 72)
    if os.name == "nt":
        print("Activate: .\\.venv\\Scripts\\Activate.ps1")
        print("Run     : .\\.venv\\Scripts\\python.exe main.py ./data/one_cr.json --provider ollama")
    else:
        print("Activate: source .venv/bin/activate")
        print("Run     : .venv/bin/python main.py ./data/one_cr.json --provider ollama")

    print("\nNotes:")
    print("- Ollama is an external application and is not installed by pip.")
    print("- The embedding model is downloaded on first use when semantic memory is enabled.")
    print("- API credentials are not created or stored by this installer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
