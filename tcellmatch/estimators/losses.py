import tensorflow as tf
import torch

class WeightedBinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):

    def __init__(
            self,
            weight_positives: float = 1,
            label_smoothing: float = 0
    ):
        """ Build instance of weighted binary crossentropy based on tf.keras.losses.BinaryCrossentropy implementation.

        :param weight_positives: Factor to multiply binary crossentropy cost of positive observation labels with.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        """
        super(WeightedBinaryCrossentropy, self).__init__(
            from_logits=True,
            label_smoothing=label_smoothing
        )
        self.weight_positives = weight_positives

    def call(self, y_true, y_pred):
        """ Computes weighted binary crossentropy loss for a batch.

        :param y_true: Observations (observations, labels).
        :param y_pred: Predictions (observations, labels).
        :return: Loss (observations, labels).
        """
        # Format data.
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # Perform label smoothing.
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute loss.
        loss = tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true,
            logits=y_pred,
            pos_weight=self.weight_positives
        )
        return loss

# src: https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
def MMD(x, y, kernel='rbf'):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
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