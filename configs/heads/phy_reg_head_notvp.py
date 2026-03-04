import torch
import torch.nn as nn
from configs.common import BaseModule


class PhyRegHeadNoTVP(BaseModule):
    """
    Ablation head: remove TVP information.
    - Interface compatible with PhyRegHead: accepts kwargs (T/V/P) but ignores them.
    - Uses only image feature x to regress k.
    """

    def __init__(self,
                 in_channels: int,
                 cond_dim: int = 3,          # keep for config compatibility; NOT used
                 hidden_dims=(512, 128),
                 dropout=0.1,
                 loss=dict(type='HuberLoss', loss_weight=1.0),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = int(in_channels)
        self.cond_dim = int(cond_dim)  # not used, kept for compatibility

        dims = [self.in_channels] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

        loss = loss or {}
        self.loss_type = loss.get('type', 'HuberLoss')
        self.loss_weight = float(loss.get('loss_weight', 1.0))
        if self.loss_type == 'HuberLoss':
            self.criterion = nn.SmoothL1Loss(reduction='mean')
        elif self.loss_type in ('MSE', 'MSELoss'):
            self.criterion = nn.MSELoss(reduction='mean')
        elif self.loss_type in ('L1Loss', 'MAE'):
            self.criterion = nn.L1Loss(reduction='mean')
        else:
            # fallback
            self.criterion = nn.SmoothL1Loss(reduction='mean')

    def _ensure_2d_feat(self, x):
        # compatible with list/tuple backbone outputs
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        if x.dim() == 4:
            x = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1)
        elif x.dim() == 3:
            x = torch.flatten(x, 1)
        return x

    def forward_train(self, x, targets, **kwargs):
        # kwargs may contain T/V/P but we ignore them in this ablation head.
        x = self._ensure_2d_feat(x)
        pred = self.mlp(x).squeeze(-1)

        if targets is None:
            raise ValueError("targets(k) is required for training.")
        tgt = targets.to(x.device).float()
        if tgt.ndim > 1:
            if tgt.shape[1] == 1:
                tgt = tgt.view(-1)
            else:
                tgt = tgt.mean(dim=1)

        loss = self.criterion(pred, tgt) * self.loss_weight
        return dict(loss=loss)

    def simple_test(self, x, **kwargs):
        # kwargs may contain T/V/P but we ignore them in this ablation head.
        x = self._ensure_2d_feat(x)
        pred = self.mlp(x).squeeze(-1)
        return pred
