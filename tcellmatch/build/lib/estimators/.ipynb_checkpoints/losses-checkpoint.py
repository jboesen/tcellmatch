import torch
import torch.nn as nn

class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, weight_positives=1.0, label_smoothing=0.0):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.weight_positives = weight_positives
        self.label_smoothing = label_smoothing

    def forward(self, y_true, y_pred):
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.weight_positives]))
        return loss_fn(y_pred, y_true)

# src: https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
def MMD(x, y, kernel='rbf'):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
        :param x: first sample, distribution P
        :param y: second sample, distribution Q
        :param kernel: kernel type such as "multiscale" or "rbf"
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)