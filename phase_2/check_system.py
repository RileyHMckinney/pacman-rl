# Save this as check_system.py in your project folder
import psutil
import os
import torch

print("=== SYSTEM DIAGNOSTICS ===")
print(f"CPU Cores: {psutil.cpu_count()} (physical: {psutil.cpu_count(logical=False)})")
print(f"RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
print(f"PyTorch using: {torch.cuda.is_available() and 'GPU' or 'CPU'}")
print(f"Current Python PID: {os.getpid()}")