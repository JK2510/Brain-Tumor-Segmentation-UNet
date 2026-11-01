import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        denominator = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        # Ignore background when calculating the overlap loss.
        return 1.0 - dice[1:].mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)
