import torch
from torch import nn


class MEC(nn.Module):
    def __init__(self, mu, lamda, n, feature_wise=False):
        super().__init__()

        self.mu = mu
        self.lamda = lamda
        self.n = n
        # If choose feature_wise, the entropy will be calculated on the feature dimension, 
        # rather than the batch dimension.
        self.feature_wise = feature_wise

    def forward(self, view1, view2):
        # view1 and view2 are two views (representations) of the same batch, with shape (B, E).
        # It's worth noting that both representations should be normalized (for example, l2 normalization).
        if self.feature_wise:
            c = self.lamda * torch.mm(view1.transpose(0, 1), view2)  # (E, E)
        else:
            c = self.lamda * torch.mm(view1, view2.transpose(0, 1))  # (B, B)
        power = c
        sum_p = torch.zeros_like(power)
        for k in range(1, self.n+1):
            if k > 1:
                power = torch.mm(power, c)
            if (k + 1) % 2 == 0:
                sum_p += power / k
            else:
                sum_p -= power / k
        loss = -1 * self.mu * torch.trace(sum_p)
        return loss
