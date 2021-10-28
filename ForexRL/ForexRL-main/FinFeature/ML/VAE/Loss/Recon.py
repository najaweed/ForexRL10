import torch.nn.functional as F
import torch


class Reconstruction:

    def __init__(self,
                 observation,
                 observation_hat
                 ):
        self.obs = observation
        self.obs_hat = observation_hat

    @property
    def mean_square(self):
        return F.mse_loss(self.obs, self.obs_hat)


loss = torch.nn.MSELoss()
input = torch.randn(30, 5, requires_grad=True)
target = torch.randn(30, 5)
output = loss(input, target)
print(output)
print(torch.clip(output, min=3, max=10))