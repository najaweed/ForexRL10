import torch


class Regularization:
    def __init__(self,
                 c_mean,
                 c_log_var,
                 ):
        self.mean = c_mean
        self.log_var = c_log_var

    @property
    def kl_div(self):
        return torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mean ** 2 - self.log_var.exp(), dim=1), dim=0)



