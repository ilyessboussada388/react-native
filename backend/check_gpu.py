import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("gpu: not available to PyTorch")