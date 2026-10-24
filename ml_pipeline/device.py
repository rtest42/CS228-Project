import platform
import torch


def get_device() -> torch.device:
    """
    Returns the device on which PyTorch is running on.
    """
    # NVIDIA CUDA or AMD ROCm
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Intel Arc
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    # Apple Silicon GPU
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")

    # CPU fallback
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(device)
