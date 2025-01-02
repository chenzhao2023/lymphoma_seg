import torch.nn as nn
from monai.losses import DiceLoss


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(to_onehot_y=False, softmax=False)
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return loss
