import torch
print(torch.cuda.is_available())
import stable_baselines3
print(stable_baselines3.common.utils.get_device(device='auto'))
