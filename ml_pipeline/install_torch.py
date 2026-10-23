import platform
import shutil
import subprocess
import sys
from typing import FInal


def detect_gpu_brand() -> str:
    """
    Detect which GPU is installed.

    Returns 'CPU' as a fallback if no GPU is found.
    """
    system: Final[str] = platform.system()

    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return "NVIDIA"

    if system == "Linux" and (
        shutil.which("rocminfo") or shutil.which("rocm-smi")
    ):
        return "AMD"

    if shutil.which("xpu-smi"):
        result = subprocess.run(
            ["xpu-smi", "discovery"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return "Intel"

    if system == "Darwin" and platform.machine() == "arm64":
        return "Apple"

    return "CPU"


def main() -> int:
    backend: Final[str] = detect_gpu_brand()
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", f"requirements-{backend}.txt"],
        check=True
    )
    return result.returncode


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
