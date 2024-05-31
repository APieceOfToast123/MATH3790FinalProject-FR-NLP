import torch
print(torch.cuda.is_available())  # 应输出 True 如果 CUDA 可用
print(torch.cuda.get_device_name(0))  # 输出你的 GPU 名称
