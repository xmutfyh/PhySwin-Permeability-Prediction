from typing import Any, Tuple, Union, Optional

import torch
import torch.nn as nn

from .reg_head import RegHead, _register_if_possible


@_register_if_possible
class LinearRegHead1(RegHead):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        # 兼容 mmcls 风格：head.loss = dict(type=..., loss_weight=..., ...)
        loss: Optional[dict] = None,
        loss_type: str = 'HuberLoss',
        loss_cfg: Optional[dict] = None,
        loss_weight: float = 1.0,
        init_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        # 解析并吞掉 loss=dict(...)，不向父类透传
        if loss is not None:
            loss_type = loss.get('type', loss_type)
            loss_weight = float(loss.get('loss_weight', loss_weight))
            loss_cfg = {k: v for k, v in loss.items() if k not in ('type', 'loss_weight')}

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            loss_type=loss_type,
            loss_cfg=loss_cfg,
            loss_weight=loss_weight,
            init_cfg=init_cfg,
            build_mlp=False,
        )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm = nn.LayerNorm(self.in_channels)
        self.fc1 = nn.Linear(self.in_channels, self.out_channels)
        self.init_weights()

    def init_weights(self) -> None:
        if hasattr(self, 'fc1'):
            nn.init.normal_(self.fc1.weight, std=0.01)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
        if hasattr(self, 'norm') and isinstance(self.norm, nn.LayerNorm) and self.norm.elementwise_affine:
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def pre_logits(self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...], list]) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[-1]
        assert isinstance(x, torch.Tensor), "Input feature must be a Tensor or tuple/list of Tensors."
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        return self.norm(x)

    def forward(self, x: Any) -> torch.Tensor:
        x = self.pre_logits(x)
        return self.fc1(x)

    def forward_train(
        self,
        x: Any,
        k: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        pred = self.forward(x)
        target = targets if targets is not None else (k if k is not None else kwargs.get('k'))
        if target is None:
            raise ValueError("k/targets is required for training.")
        target = target.to(pred.device, dtype=pred.dtype)
        pred_for_loss = pred.squeeze(-1) if (pred.ndim >= 2 and pred.size(-1) == 1) else pred
        if target.ndim > 1:
            if target.ndim == 2 and target.shape[-1] == 1:
                target = target.view(-1)
            else:
                target = target.view(target.size(0), -1).mean(dim=1)
        loss = self.compute_loss(pred_for_loss, target)
        return dict(loss=loss, pred=pred)

    def simple_test(self, x: Any, post_process: bool = False, **kwargs):
        pred = self.forward(x)
        if post_process:
            return pred.detach().cpu().view(pred.size(0), -1).tolist()
        return pred
