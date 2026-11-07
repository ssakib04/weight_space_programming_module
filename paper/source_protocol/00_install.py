# ---------------- Install (Kaggle: Internet ON) ----------------
# Only packages that Kaggle's image may lack/outdate. torch/torchvision/numpy/pandas are preinstalled; never reinstall torch.
import os, subprocess, sys
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "anthropic>=0.40", "bitsandbytes>=0.43", "accelerate>=0.33"],
                       capture_output=True, text=True)
    print("pip rc", r.returncode, r.stderr[-800:])
