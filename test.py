import torch
try:
    torch.cuda.current_device()
    print("CUDA is initialized successfully!")
except Exception as e:
    print(f"CUDA initialization error: {e}")
