import torch
import torch.nn as nn
from configs.common import BaseModule


class PhyRegHead(BaseModule):
    """
    Regression head with physics/environment condition (T, V, P).

    Structure (match diagram):
      1) Condition branch: (T,V,P) -> FC -> FC -> FC  (default 3-layer MLP)
         - configurable by cond_hidden_dims, any length >= 1
      2) Fuse: concat(image_feat, cond_feat)
      3) Regressor MLP -> scalar output
         - configurable by hidden_dims (can be empty to make it a single Linear after concat)
    """

    def __init__(
        self,
        in_channels: int,
        cond_dim: int = 3,
        # TVP branch before fusion: default 3 FC layers to match the diagram
        cond_hidden_dims=(64, 64, 64),
        # fused regressor hidden dims after concatenation
        hidden_dims=(512, 128),
        dropout=0.1,
        loss=dict(type='HuberLoss', loss_weight=1.0),
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = int(in_channels)
        self.cond_dim = int(cond_dim)

        # ---------- Condition branch (N FC layers, default 3) ----------
        cond_hidden_dims = list(cond_hidden_dims) if cond_hidden_dims is not None else []
        if len(cond_hidden_dims) < 1:
            raise ValueError(f'cond_hidden_dims must have at least 1 element (got {cond_hidden_dims}).')

        cond_layers = []
        in_dim = self.cond_dim
        for h in cond_hidden_dims:
            h = int(h)
            cond_layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                cond_layers += [nn.Dropout(float(dropout))]
            in_dim = h
        self.cond_mlp = nn.Sequential(*cond_layers)
        self.cond_out_dim = in_dim  # last hidden dim

        # ---------- Fused regressor after concat ----------
        fused_in = self.in_channels + self.cond_out_dim
        hidden_dims = list(hidden_dims) if hidden_dims is not None else []
        dims = [fused_in] + hidden_dims + [1]

        fused_layers = []
        # if hidden_dims empty -> single Linear(fused_in -> 1)
        for i in range(len(dims) - 2):
            fused_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                fused_layers += [nn.Dropout(float(dropout))]
        fused_layers += [nn.Linear(dims[-2], dims[-1])]
        self.mlp = nn.Sequential(*fused_layers)

        # ---------- Loss ----------
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
            self.criterion = nn.SmoothL1Loss(reduction='mean')

    def _stack_cond(self, kwargs, device, batch_size: int):
        """Collect T/V/P from kwargs and shape into [B, cond_dim]."""
        vals = []
        for key in ('T', 'V', 'P'):
            v = kwargs.get(key, None)
            if v is None:
                continue
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            v = v.to(device).float()
            if v.ndim == 1:
                v = v.view(batch_size, 1)
            else:
                v = v.view(batch_size, -1)
            vals.append(v)

        if len(vals) == 0:
            cond = torch.zeros(batch_size, self.cond_dim, device=device)
        else:
            cond = torch.cat(vals, dim=1)

        # pad/crop to cond_dim
        if cond.shape[1] < self.cond_dim:
            pad = torch.zeros(batch_size, self.cond_dim - cond.shape[1], device=device)
            cond = torch.cat([cond, pad], dim=1)
        elif cond.shape[1] > self.cond_dim:
            cond = cond[:, :self.cond_dim]
        return cond

    def _ensure_2d_feat(self, x):
        """Make sure image features are [B, C]."""
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        if x.dim() == 4:
            # [B,C,H,W] -> GAP -> [B,C]
            x = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1)
        elif x.dim() == 3:
            x = torch.flatten(x, 1)
        return x

    def _forward_pred(self, x, **kwargs):
        x = self._ensure_2d_feat(x)
        B = x.shape[0]
        cond_raw = self._stack_cond(kwargs, x.device, B)
        cond_feat = self.cond_mlp(cond_raw)
        fused = torch.cat([x, cond_feat], dim=1)
        pred = self.mlp(fused).squeeze(-1)
        return pred

    def forward_train(self, x, targets, **kwargs):
        pred = self._forward_pred(x, **kwargs)

        if targets is None:
            raise ValueError('targets(k) is required for training.')
        tgt = targets.to(pred.device).float()
        if tgt.ndim > 1:
            if tgt.shape[1] == 1:
                tgt = tgt.view(-1)
            else:
                tgt = tgt.mean(dim=1)

        loss = self.criterion(pred, tgt) * self.loss_weight
        return dict(loss=loss)

    def simple_test(self, x, **kwargs):
        return self._forward_pred(x, **kwargs)
