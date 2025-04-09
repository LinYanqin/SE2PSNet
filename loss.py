import torch
from torch import nn
def loss_fun(output, target):
    epsilon = 1e-8

    output = output.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)

    loss_fun_strong = nn.MSELoss()
    loss_strong = loss_fun_strong(output[..., 0], target[..., 2])

    absolute_error = torch.abs(output[..., 0] - target[..., 2])

    relative_error = absolute_error / (torch.abs(target[..., 2]) + epsilon)

    loss = torch.mean(relative_error) + loss_strong

    return loss

