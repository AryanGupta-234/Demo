#!/usr/bin/env python3
"""
One-command installer for the Pre-CAB Validator.

Usage:
    python install.py
    python install.py --full

Default mode installs only what is needed for the offline manager demo:
document extraction, synthetic PDF generation, and the core validator.

--full installs the complete development/AI/API dependency set and runs
the full pytest suite.
"""

from __future__ import annotations

import argparse
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


def activate_windows_terminal() -> None:
    """Open a new Windows CMD session with the project venv activated.

    A child process cannot modify the environment of the CMD/PowerShell process
    that launched this installer. Therefore the reliable automatic behavior is
    to open a new CMD window rooted at the repository with activate.bat run.
    """
    if os.name != "nt":
        return

    activate_bat = VENV / "Scripts" / "activate.bat"
    if not activate_bat.exists():
        return

    print("\nOpening a new CMD window with .venv activated...")
    command = f'cd /d "{ROOT}" && call "{activate_bat}"'
    subprocess.Popen(
        ["cmd.exe", "/k", command],
        cwd=ROOT,
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Pre-CAB Validator")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Install the complete AI/API/development stack and run all tests.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Pre-CAB Validator — One-Command Installer")
    print("=" * 72)
    print(f"Platform : {platform.system()} {platform.machine()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Project  : {ROOT}")
    print(f"Mode     : {'FULL' if args.full else 'MANAGER DEMO'}")

    if sys.version_info < (3, 11):
        print("\nERROR: Python 3.11 or newer is required.")
        print("Install Python 3.11+ and run this installer again.")
        return 1

    if not (ROOT / "pyproject.toml").exists():
        print("\nERROR: pyproject.toml was not found.")
        print("Run this installer from the repository root.")
        return 1

    if not VENV.exists():
        print("\n[1/5] Creating isolated virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV)])
    else:
        print("\n[1/5] Existing .venv found — reusing it.")

    py = venv_python()
    if not py.exists():
        print(f"ERROR: Virtual-environment Python was not found: {py}")
        return 1

    print("\n[2/5] Upgrading packaging tools...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    extras = "all,dev" if args.full else "docs"
    print(f"\n[3/5] Installing {('full' if args.full else 'manager-demo')} dependencies...")
    run([str(py), "-m", "pip", "install", "-e", f".[{extras}]"])

    print("\n[4/5] Verifying the installation...")
    run([str(py), "-m", "compileall", "-q", "src", "scripts", "main.py"])

    if args.full:
        run([str(py), "-m", "pytest"])
    else:
        run([str(py), "scripts/setup_manager_demo.py"])
        run([str(py), "scripts/demo.py"])

    print("\n[5/5] Finalizing...")
    print("\n" + "=" * 72)
    print("INSTALLATION + DEMO SETUP COMPLETE")
    print("=" * 72)
    if os.name == "nt":
        print("Activate : .\\.venv\\Scripts\\Activate.ps1")
        print("CMD      : .\\.venv\\Scripts\\activate")
        print("Demo     : .\\.venv\\Scripts\\python.exe scripts\\demo.py")
        print("Full     : .\\install.py --full")
        print("Workspace: .\\data\\manager_demo")
    else:
        print("Activate : source .venv/bin/activate")
        print("Demo     : .venv/bin/python scripts/demo.py")
        print("Full     : .venv/bin/python install.py --full")
        print("Workspace: ./data/manager_demo")

    print("\nNotes:")
    print("- Default mode is intentionally lightweight and does not install PyTorch/sentence-transformers.")
    print("- --full installs semantic embeddings, LLM/API clients, and development/test dependencies.")
    print("- Ollama is an external application and is not installed by pip.")
    print("- The embedding model is downloaded on first use when semantic memory is enabled.")
    print("- OCR Python packages are installed automatically; Tesseract itself must be installed on the machine (or set TESSERACT_CMD).")
    print("- Set PRE_CAB_VISION_MODEL to a local Ollama vision-capable model to enable image vision analysis.")
    print("- API credentials are not created or stored by this installer.")
    print("- Ruff is optional developer tooling and is not required for installation or tests.")
    print("- Manager demo data and generated evidence are synthetic and contain no production data.")

    activate_windows_terminal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
