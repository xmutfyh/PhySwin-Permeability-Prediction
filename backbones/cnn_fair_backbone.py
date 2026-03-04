# configs/backbones/cnn_fair_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 你的工程里 BuildNet 继承 BaseModule，所以 backbone 也尽量继承它
    from configs.common import BaseModule
except Exception:
    BaseModule = nn.Module


class CNNFairBackbone(BaseModule):
    """
    公平对比版 CNN Backbone（可替换 Swin 的 backbone；
    TVP 可在 backbone 内 FiLM 融合，也可交给回归头 PhyRegHead 融合）

    设计目标：
    - 输入: [B,3,224,224]
    - 输出: [B, out_channels, 7, 7]（默认 out_channels=768，7x7 对齐 Swin-Tiny 最后一层特征分辨率）
    - 深度对齐：num_blocks=(2,2,6,2) 对齐 Swin-Tiny depths

    TVP 融合：
    - 当 use_tvp=True 时，在 stage4 输出后用 FiLM（gamma/beta）调制特征
    - 如果你采用 PhyRegHead 来融合 TVP（与 Swin 一致的 head late-fusion），建议在 config 中设 use_tvp=False，
      避免 TVP 在 backbone+head 被“双重融合”。
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        out_channels: int = 768,
        num_blocks=(2, 2, 6, 2),
        # ===== TVP 相关 =====
        use_tvp: bool = False,
        tvp_dim: int = 3,
        tvp_hidden: int = 128,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        self.use_tvp = bool(use_tvp)
        self.tvp_dim = int(tvp_dim)
        self.tvp_hidden = int(tvp_hidden)

        # ===== Stem =====
        # 224 -> 112
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        # 112 -> 56
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ===== Stages =====
        # stage1: 56x56, C=base
        self.stage1 = self._make_stage(base_channels, base_channels, num_blocks[0], stride=1)
        # stage2: 56->28, C=base*2
        self.stage2 = self._make_stage(base_channels, base_channels * 2, num_blocks[1], stride=2)
        # stage3: 28->14, C=base*4
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, num_blocks[2], stride=2)
        # stage4: 14->7, C=out_channels（默认 768）
        self.stage4 = self._make_stage(base_channels * 4, out_channels, num_blocks[3], stride=2)

        # ===== TVP FiLM 融合层 =====
        self.tvp_mlp = None
        if self.use_tvp:
            # 生成 [gamma, beta] 各 C 维，所以输出 2C
            self.tvp_mlp = nn.Sequential(
                nn.Linear(self.tvp_dim, self.tvp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.tvp_hidden, 2 * out_channels),
            )

    def _make_stage(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride=stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    @staticmethod
    def _ensure_2d(x, name: str):
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        elif x.dim() == 0:
            x = x.view(1, 1)
        return x

    def _film_fuse(self, feat: torch.Tensor, T, V, P) -> torch.Tensor:
        """
        feat: [B,C,H,W]
        T/V/P: [B] or [B,1]
        """
        if self.tvp_mlp is None:
            return feat

        T = self._ensure_2d(T, "T")
        V = self._ensure_2d(V, "V")
        P = self._ensure_2d(P, "P")
        if T is None or V is None or P is None:
            raise ValueError("use_tvp=True 但 forward 未提供 T/V/P")

        tvp = torch.cat([T, V, P], dim=1).to(feat.device)  # [B,3]
        gb = self.tvp_mlp(tvp)  # [B, 2C]
        gamma, beta = torch.chunk(gb, 2, dim=1)  # [B,C], [B,C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B,C,1,1]

        # FiLM：x' = x*(1+gamma) + beta
        feat = feat * (1.0 + gamma) + beta
        return feat

    def forward(self, x: torch.Tensor, T=None, V=None, P=None, **kwargs) -> torch.Tensor:
        """
        x: [B,3,224,224]
        返回: [B,768,7,7]
        """
        x = self.stem(x)      # [B, base, 112, 112]
        x = self.pool(x)      # [B, base, 56, 56]

        x = self.stage1(x)    # [B, base, 56, 56]
        x = self.stage2(x)    # [B, base*2, 28, 28]
        x = self.stage3(x)    # [B, base*4, 14, 14]
        x = self.stage4(x)    # [B, 768, 7, 7]

        # ✅ TVP 融合（兼容 T/V/P 也可能通过 kwargs 传入）
        if self.use_tvp:
            if T is None:
                T = kwargs.get("T", None)
            if V is None:
                V = kwargs.get("V", None)
            if P is None:
                P = kwargs.get("P", None)
            x = self._film_fuse(x, T, V, P)

        return x


class BasicBlock(nn.Module):
    """
    非瓶颈残差块（轻量、稳定），避免过度“堆花活”影响公平性
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out
