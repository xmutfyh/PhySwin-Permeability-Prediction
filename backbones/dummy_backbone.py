import torch
import torch.nn as nn
from configs.common import BaseModule

class DummyBackbone(BaseModule):
    def __init__(self, out_channels=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.out_channels = int(out_channels)

    def forward(self, x, **kwargs):
        # 返回一个极小的特征，占位即可；后续 head 会忽略
        if x is None:
            B = 1
            device = "cpu"
        else:
            B = x.shape[0]
            device = x.device
        return torch.zeros(B, self.out_channels, 1, 1, device=device)
