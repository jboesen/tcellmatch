import torch
from torch import Tensor

def custom_r2(y_true : Tensor, y_pred : Tensor):
    """
    :param y_true:
    :param y_pred:
    :return: (graphs,)
    """
    r2 = 1. - torch.sum(torch.square(y_true - y_pred)) / \
         torch.sum(torch.square(y_true - torch.mean(y_true)))
    return torch.mean(r2)


def custom_logr2(y_true : Tensor, y_pred : Tensor):
    """
    :param y_true:
    :param y_pred:
    :return: (graphs,)
    """
    eps = 1.
    y_true = torch.log(y_true + eps)
    y_pred = torch.log(y_pred + eps)
    return custom_r2(y_true=y_true, y_pred=y_pred)
