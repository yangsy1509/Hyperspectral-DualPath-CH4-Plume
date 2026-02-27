import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, from_logits=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.from_logits = from_logits

    def forward(self, preds, targets):
        if self.from_logits:
            preds = torch.sigmoid(preds)

        preds = preds.view(-1)
        targets = targets.view(-1)

        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()

        score = (TP + 1e-7) / (TP + self.alpha * FN + self.beta * FP + 1e-7)
        return 1 - score