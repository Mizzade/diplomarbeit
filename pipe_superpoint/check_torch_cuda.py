import torch

print('Device: ', torch.cuda.current_device())

print('Device address: ', torch.cuda.device(0))

print('Number of devices: ', torch.cuda.device_count())

print('Device name: ', torch.cuda.get_device_name(0))
