import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA build version:", torch.version.cuda)
print("Device Name:", torch.cuda.get_device_name(0))
  