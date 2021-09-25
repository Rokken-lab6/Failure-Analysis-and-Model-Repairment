# -*- coding: utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sumagg.sumagg import SumAgg

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()

        self.cfg = cfg

    def forward(self, y_pred_dict, pos):
        # y_pred: B, T, C, H, W
        # pos: list of gt

        logits = y_pred_dict.logits
        gt = pos

        B, T, N_classes = logits.shape
        logits = logits.view(B*T, N_classes)

        gt = gt.cuda().long().view(B*T)
        loss_val = nn.CrossEntropyLoss()(logits, gt)

        SumAgg().add("loss_val", loss_val.item())
        return loss_val


    def predict(self, y):

        logits = y.logits
        
        preds = logits.argmax(-1).cpu().numpy()

        return preds