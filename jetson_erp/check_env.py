"""Quick sanity check of a Jetson (or PC) environment for the ERP challenge.

Prints versions, GPU availability, memory, and a tiny FP32/FP16 matmul benchmark.
Usage: python check_env.py
"""
import platform
import time

print("python   ", platform.python_version(), platform.machine())
try:
    with open("/etc/nv_tegra_release") as f:
        print("L4T      ", f.readline().strip())
except OSError:
    print("L4T       (not a Jetson)")

import numpy as np
import cv2
print("numpy    ", np.__version__)
print("opencv   ", cv2.__version__, "| CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0)

try:
    import torch
    print("torch    ", torch.__version__, "| cuda", torch.cuda.is_available(), "| cuDNN", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        print("gpu      ", torch.cuda.get_device_name(0), "| sm", torch.cuda.get_device_capability(0))
        for dt in (torch.float32, torch.float16):
            x = torch.randn(2048, 2048, device="cuda", dtype=dt)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(20):
                y = x @ x
            torch.cuda.synchronize(); dt_s = (time.perf_counter() - t) / 20
            print(f"matmul    {str(dt):14s} {2 * 2048 ** 3 / dt_s / 1e12:6.2f} TFLOPS")
except ImportError:
    print("torch     NOT INSTALLED")

for mod in ("tensorrt", "lightglue", "lpips", "skimage", "scipy"):
    try:
        m = __import__(mod)
        print(f"{mod:9s}", getattr(m, "__version__", "ok"))
    except ImportError:
        print(f"{mod:9s} NOT INSTALLED")

try:
    with open("/proc/meminfo") as f:
        mem = {l.split(":")[0]: l.split(":")[1].strip() for l in f}
    print("memory   ", mem.get("MemTotal"), "| swap", mem.get("SwapTotal"))
except OSError:
    pass
