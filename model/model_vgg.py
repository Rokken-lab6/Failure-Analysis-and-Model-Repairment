# -*- coding: utf8 -*-

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_repair.tasks.extract_checkpoint import FeaturesSaver

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.cfg = cfg

        self.model = make_layers([16, 'M', 32, 'M', 128, 128, 'M', 256, 256], batch_norm=False, in_channels=cfg.n_c_input)
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=cfg.n_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x_BT, state=None):
        B, T, C, H, W = x_BT.shape

        if self.cfg.max_normalize:
            x_BT = x_BT / 255.0

        x_B = x_BT.view(B*T, C, H, W)
        
        h_B = x_B
        for layer in self.model:
            h_B = layer(h_B)
            FeaturesSaver().save(str(layer), h_B)

        y_B = self.final_conv(h_B) # Bx3xHxW

        y_B = y_B.mean(-1).mean(-1) # B*TxN_class

        N_class = y_B.shape[1]
        y_BT = y_B.view(B, T, N_class)
        

        Output = namedtuple('Output', ['logits', 'state'])
        y = Output(logits=y_BT, state=None)

        return y


def make_layers(cfg, batch_norm, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
