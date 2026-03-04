from configs.backbones import *
from configs.necks import *
from configs.heads import *
from importlib import import_module

from configs.heads.linear_reghead3 import LinearRegHead3
from configs.heads.reg_head import RegHead
from configs.backbones.resnet import ResNet
from configs.necks.gap import GlobalAveragePooling
from configs.common import BaseModule, Sequential
import torch.optim as optim
import torch.nn as nn
import torch

from configs.heads.headcnn import CNNRegHead
from core.optimizers.lr_update import StepLrUpdater, LrUpdater, PolyLrUpdater, CosineAnnealingLrUpdater, \
    CosineAnnealingCooldownLrUpdater, ReduceLROnPlateauLrUpdater
from configs.heads.linear_reghead2 import LinearRegHead2
from configs.heads.phy_reg_head import PhyRegHead
from configs.heads.phy_temporal_head import PhyTemporalHead


def build_model(cfg):
    if isinstance(cfg, list):
        modules = [eval(cfg_.pop("type"))(**cfg_) for cfg_ in cfg]
        return Sequential(*modules)
    else:
        cfg = cfg.copy()
        return eval(cfg.pop("type"))(**cfg)


def _take_first_if_seq(x):
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return x[0]
    return x


class BuildNet(BaseModule):
    def __init__(self, cfg):
        super(BuildNet, self).__init__()
        self.neck_cfg = cfg.get("neck")
        self.head_cfg = cfg.get("head")
        self.backbone = build_model(cfg.get("backbone"))
        self.neck = build_model(cfg.get("neck")) if self.neck_cfg is not None else None
        self.head = build_model(cfg.get("head")) if self.head_cfg is not None else None

    def freeze_layers(self, names):
        assert isinstance(names, tuple)
        for name in names:
            if not hasattr(self, name):
                continue
            layers = getattr(self, name)
            for p in layers.parameters():
                p.requires_grad = False

    def extract_feat(self, img, stage='neck', **kwargs):
        """
        支持：
        - 4D: [B,C,H,W]
        - 5D: [B,T,C,H,W]

        关键修复点：
        ✅ 将 T/V/P 从 kwargs 透传给 backbone（如果 backbone 支持）
        """
        if img is None:
            raise ValueError("Input img is None")

        # 只取 TVP，避免污染其它模块
        tvp_kwargs = {}
        for k in ("T", "V", "P"):
            if k in kwargs and kwargs[k] is not None:
                tvp_kwargs[k] = kwargs[k]

        # ===== 5D 输入 =====
        if isinstance(img, torch.Tensor) and img.dim() == 5:
            B, Tt, C, H, W = img.shape
            x = img.view(B * Tt, C, H, W)

            try:
                x = self.backbone(x, **tvp_kwargs)
            except TypeError:
                x = self.backbone(x)

            x = _take_first_if_seq(x)

            if stage == 'backbone':
                if isinstance(x, torch.Tensor):
                    return x.view(B, Tt, *x.shape[1:])
                return x

            if self.neck is not None:
                x = self.neck(x)
                x = _take_first_if_seq(x)

            if isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[-1] == 1 and x.shape[-2] == 1:
                x = x.flatten(1)

            if isinstance(x, torch.Tensor) and x.dim() == 2:
                x = x.view(B, Tt, x.shape[-1])
            else:
                x = x.view(B, Tt, *x.shape[1:])

            return x

        # ===== 4D 输入（你当前 CNN / Swin 都走这里）=====
        try:
            x = self.backbone(img, **tvp_kwargs)
        except TypeError:
            x = self.backbone(img)

        x = _take_first_if_seq(x)

        if stage == 'backbone':
            return x

        if self.neck is not None:
            x = self.neck(x)
            x = _take_first_if_seq(x)

        return x

    def forward(self, x, return_loss=True, train_statu=False, **kwargs):
        if x is None:
            raise ValueError("Input x is None")

        # 🔥 关键：把 kwargs 传进 extract_feat
        feats = self.extract_feat(x, **kwargs)

        if train_statu:
            return self.forward_test(feats, **kwargs), self.forward_train(feats, **kwargs)

        if return_loss:
            return self.forward_train(feats, **kwargs)
        else:
            return self.forward_test(feats, **kwargs)

    def forward_train(self, x, targets=None, **kwargs):
        if self.head is None:
            raise RuntimeError('Head is not defined.')
        return self.head.forward_train(x, targets, **kwargs)

    def forward_test(self, x, **kwargs):
        if self.head is None:
            raise RuntimeError('Head is not defined.')
        return self.head.simple_test(x, **kwargs)
