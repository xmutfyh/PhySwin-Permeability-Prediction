import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.common import BaseModule


def _adaptive_pool_2d(x):
    # x: [*, C, H, W] -> [*, C]
    return F.adaptive_avg_pool2d(x, output_size=1).flatten(-3)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, dropout=0.1):
        super().__init__()
        pad = d * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):  # x: [B, F, T]
        return self.net(x)


class TemporalUnit(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, mode='gru', num_layers=1,
                 dropout=0.1, tcn_layers=3):
        super().__init__()
        self.mode = mode.lower()
        if self.mode == 'gru':
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.out_dim = hidden_dim
        elif self.mode == 'tcn':
            layers = []
            f = in_dim
            for i in range(tcn_layers):
                d = 2 ** i
                layers.append(
                    TCNBlock(
                        in_ch=f,
                        out_ch=hidden_dim,
                        d=d,
                        dropout=dropout
                    )
                )
                f = hidden_dim
            self.tcn = nn.Sequential(*layers)
            self.out_dim = hidden_dim
        else:
            raise ValueError(f'Unknown temporal mode: {mode}')

    def forward(self, x):  # x: [B, T, F]
        if self.mode == 'gru':
            out, _ = self.rnn(x)  # [B, T, H]
            return out
        else:
            out = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # [B, T, H]
            return out


class PhyTemporalHead(BaseModule):
    """
    K(t) 回归 + CRP/FRP 分类（多任务）——兼容视觉 + 可选 TVP 条件：
      - 支持输入形状: [B,T,C], [B,C], [B,T,C,H,W], [B,C,H,W]
      - 当 cond_dim > 0 时：使用 (T,V,P,...) 条件
      - 当 cond_dim = 0 时：完全忽略条件，只用视觉特征
    """
    def __init__(self,
                 in_channels,
                 cond_dim=3,
                 cond_embed=64,
                 feat_embed=256,
                 temporal='gru',
                 temporal_hidden=256,
                 temporal_layers=1,
                 tcn_layers=3,
                 dropout=0.1,
                 num_classes=2,
                 # losses
                 loss=dict(type='HuberLoss', delta=1.0),
                 reg_loss_weight=1.0,
                 cls_loss_weight=1.0,
                 # 单调约束（默认对 V 正单调）
                 monotonic_cfg=dict(
                     enabled=False,
                     weight=0.0,
                     eps=0.05,
                     margin=0.0,
                     cond_order=['T', 'V', 'P'],
                     pos_dims_names=['V'],
                     neg_dims_names=[]
                 ),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.cond_embed = cond_embed
        self.feat_embed = feat_embed
        self.dropout = dropout
        self.num_classes = num_classes
        self.delta = float(loss.get('delta', 1.0))
        self.reg_loss_weight = float(reg_loss_weight)
        self.cls_loss_weight = float(cls_loss_weight)
        self.monotonic_cfg = monotonic_cfg or dict(enabled=False, weight=0.0)

        # 视觉特征投影
        self.feat_proj = nn.Sequential(
            nn.Linear(in_channels, feat_embed),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 是否使用 TVP 条件（cond_dim=0 表示完全不用）
        self.use_cond = (cond_dim > 0)

        if self.use_cond:
            # 条件 (T,V,P,...) 投影
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, cond_embed),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            temporal_input_dim = feat_embed + cond_embed
        else:
            # 不使用条件；只用视觉特征
            self.cond_proj = None
            temporal_input_dim = feat_embed

        # 时序单元（GRU 或 TCN）
        self.temporal = TemporalUnit(
            in_dim=temporal_input_dim,
            hidden_dim=temporal_hidden,
            mode=temporal,
            num_layers=temporal_layers,
            dropout=dropout,
            tcn_layers=tcn_layers
        )

        # 回归头：K(t)
        self.regressor = nn.Sequential(
            nn.Linear(self.temporal.out_dim, self.temporal.out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.temporal.out_dim // 2, 1)
        )

        # 分类头：阶段 CRP/FRP
        self.classifier = nn.Linear(self.temporal.out_dim, num_classes)

        # Huber 损失（SmoothL1）
        self._huber = nn.SmoothL1Loss(
            reduction='none',
            beta=self.delta
        )  # PyTorch >= 1.10

    # 将任意输入统一为 [B, T, F] 特征序列
    def _to_seq_feats(self, x):
        if x.dim() == 5:
            # [B,T,C,H,W] -> GAP -> [B,T,C]
            B, T, C, H, W = x.shape
            f = _adaptive_pool_2d(x.view(B * T, C, H, W)).view(B, T, C)
            return f
        elif x.dim() == 4:
            # [B,C,H,W] -> GAP -> [B,1,C]
            B, C, H, W = x.shape
            f = _adaptive_pool_2d(x).view(B, 1, C)
            return f
        elif x.dim() == 3:
            # [B,T,C] 已池化
            return x
        elif x.dim() == 2:
            # [B,C] 已池化
            return x.view(x.size(0), 1, x.size(1))
        else:
            raise ValueError(f'Unsupported input shape: {tuple(x.shape)}')

    def _pack_cond(self, T, V, P, B, Tlen, device, dtype=torch.float32):
        """
        将 (T,V,P) 打包为 [B, T, cond_dim]。
        - 当 self.use_cond=False 时，直接返回 None。
        """
        if not self.use_cond:
            return None

        stacks = []
        for t in (T, V, P):
            if t is None:
                stacks.append(
                    torch.zeros(
                        (B, Tlen, 1),
                        device=device,
                        dtype=dtype
                    )
                )
            else:
                tt = t.to(device=device, dtype=dtype).view(B, Tlen, -1)
                stacks.append(tt)
        return torch.cat(stacks, dim=-1)  # [B, T, cond_dim]

    def _monotonic_penalty(self, pred_k, T, V, P):
        """
        单调性约束（默认对 V 正单调）。
        与是否使用 cond_dim 无强绑定，只依赖传入的 T/V/P。
        """
        cfg = self.monotonic_cfg or {}
        if not cfg.get('enabled', False):
            return pred_k.new_zeros(())

        eps = float(cfg.get('eps', 0.05))
        w = float(cfg.get('weight', 0.0))
        if w <= 0 or pred_k.size(1) < 2:
            return pred_k.new_zeros(())

        penalty = pred_k.new_zeros(())
        pos_names = set(cfg.get('pos_dims_names', []))

        def pen_one(cond):
            if cond is None or cond.size(1) < 2:
                return pred_k.new_zeros(())
            dv = cond[:, 1:] - cond[:, :-1]      # [B, T-1, 1]
            dk = pred_k[:, 1:] - pred_k[:, :-1]  # [B, T-1, 1]
            mask = (dv.squeeze(-1) > eps)
            if mask.any():
                p = F.relu(-dk.squeeze(-1)[mask])  # 要求 dk >= 0
                return p.mean() if p.numel() > 0 else pred_k.new_zeros(())
            return pred_k.new_zeros(())

        if 'V' in pos_names and V is not None:
            penalty = penalty + pen_one(V)

        return w * penalty

    def forward_train(self,
                      x,
                      targets=None,
                      T=None,
                      V=None,
                      P=None,
                      stage=None,
                      k_mask=None,
                      **kwargs):
        """
        x: [B,T,C]/[B,C] 或 [B,T,C,H,W]/[B,C,H,W]
        targets: [B,T,1]/[B,T]/[B,1] float
        stage: [B,T] long (0=CRP,1=FRP)，可缺省
        """
        feats = self._to_seq_feats(x)  # [B, T, C]
        B, Tlen, C = feats.shape
        device = feats.device

        # 视觉特征
        feats = self.feat_proj(feats.view(B * Tlen, C)).view(B, Tlen, -1)

        # 条件特征（TVP），在 use_cond=False 时直接忽略
        cond = self._pack_cond(T, V, P, B, Tlen, device)

        if self.use_cond:
            cond = self.cond_proj(cond)                # [B, T, cond_embed]
            z = torch.cat([feats, cond], dim=-1)       # [B, T, F'+cond]
        else:
            z = feats                                   # [B, T, F']

        # 时序建模
        z = self.temporal(z)                           # [B, T, H]

        # 回归 + 分类
        pred = self.regressor(z).squeeze(-1)           # [B, T]
        logits = self.classifier(z)                    # [B, T, num_classes]

        losses = {}
        total = pred.new_zeros(())

        # 回归监督 K(t)
        if targets is not None:
            tgt = targets.to(device=device, dtype=pred.dtype).view(B, Tlen)
            reg_loss_map = self._huber(pred, tgt)      # [B, T]

            if k_mask is not None:
                m = k_mask.to(device=device, dtype=pred.dtype).view(B, Tlen)
                denom = m.sum().clamp_min(1.0)
                loss_sup = (reg_loss_map * m).sum() / denom
                per_sample = (reg_loss_map * m).view(-1)
            else:
                loss_sup = reg_loss_map.mean()
                per_sample = reg_loss_map.view(-1)

            losses['loss_sup'] = loss_sup
            losses['per_sample_sup'] = per_sample
            total = total + self.reg_loss_weight * loss_sup
        else:
            losses['per_sample_sup'] = pred.new_zeros((B * Tlen,))

        # 阶段分类监督
        if stage is not None and self.cls_loss_weight > 0:
            st = stage.to(device=device, dtype=torch.long).view(B, Tlen)
            cls_loss = F.cross_entropy(
                logits.view(B * Tlen, -1),
                st.view(-1),
                ignore_index=-100
            )
            losses['loss_cls'] = cls_loss
            total = total + self.cls_loss_weight * cls_loss

        # 单调约束（可选）
        mono = self._monotonic_penalty(pred.unsqueeze(-1), T, V, P)
        if mono.numel() == 1 and mono > 0:
            losses['loss_mono'] = mono
            total = total + mono

        losses['loss'] = total
        return losses

    def simple_test(self,
                    x,
                    T=None,
                    V=None,
                    P=None,
                    return_stage=False,
                    **kwargs):
        """
        推理接口：
          - return_stage=False: 返回 pred_k 序列
          - return_stage=True : 返回 (pred_k, stage_probs)
        """
        feats = self._to_seq_feats(x)  # [B, T, C]
        B, Tlen, C = feats.shape
        device = feats.device

        feats = self.feat_proj(feats.view(B * Tlen, C)).view(B, Tlen, -1)

        cond = self._pack_cond(T, V, P, B, Tlen, device)

        if self.use_cond:
            cond = self.cond_proj(cond)
            z = torch.cat([feats, cond], dim=-1)
        else:
            z = feats

        z = self.temporal(z)
        pred = self.regressor(z).squeeze(-1)  # [B, T]

        if return_stage:
            logits = self.classifier(z)        # [B, T, num_classes]
            probs = logits.softmax(dim=-1)
            return pred, probs

        return pred
