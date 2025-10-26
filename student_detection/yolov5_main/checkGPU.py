import torch

#check gpu to make sure gpu is being used not cpu

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))