import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use GPU.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. PyTorch is using CPU.")
