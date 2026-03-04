import torch.nn as nn
import torch.nn.functional as F

# 定义适用于回归任务的归一化层
def BN(*args, **kwargs):
    """返回 BatchNorm2d 层"""
    return nn.BatchNorm2d(*args, **kwargs)

def BN1d(*args, **kwargs):
    """返回 BatchNorm1d 层"""
    return nn.BatchNorm1d(*args, **kwargs)

def BN2d(*args, **kwargs):
    """返回 BatchNorm2d 层"""
    return nn.BatchNorm2d(*args, **kwargs)

def BN3d(*args, **kwargs):
    """返回 BatchNorm3d 层"""
    return nn.BatchNorm3d(*args, **kwargs)

def SyncBN(*args, **kwargs):
    """返回 SyncBatchNorm 层"""
    return nn.SyncBatchNorm(*args, **kwargs)

def GN(num_groups, num_channels, **kwargs):
    """返回 GroupNorm 层"""
    return nn.GroupNorm(num_groups, num_channels, **kwargs)

def LN(normalized_shape, **kwargs):
    """返回 LayerNorm 层"""
    return nn.LayerNorm(normalized_shape, **kwargs)

def IN(*args, **kwargs):
    """返回 InstanceNorm2d 层"""
    return nn.InstanceNorm2d(*args, **kwargs)

def IN1d(*args, **kwargs):
    """返回 InstanceNorm1d 层"""
    return nn.InstanceNorm1d(*args, **kwargs)

def IN2d(*args, **kwargs):
    """返回 InstanceNorm2d 层"""
    return nn.InstanceNorm2d(*args, **kwargs)

def IN3d(*args, **kwargs):
    """返回 InstanceNorm3d 层"""
    return nn.InstanceNorm3d(*args, **kwargs)

class LayerNorm2d(nn.LayerNorm):
    """2D 图像的 LayerNorm，对通道进行归一化。

    参数:
        num_channels (int): 输入张量的通道数。
        eps (float): 为了数值稳定性，分母中添加的值。默认值为 1e-5。
        elementwise_affine (bool): 当设置为 ``True`` 时，模块将有可学习的每个元素的仿射参数，
                                   初始化为 1（权重）和 0（偏差）。默认值为 True。
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(num_channels, eps=eps, elementwise_affine=elementwise_affine)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        assert x.dim() == 4, f'LayerNorm2d 只支持形状为 (N, C, H, W) 的输入，但得到了形状为 {x.shape} 的张量'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)
        return x

def LN2d(*args, **kwargs):
    """返回 LayerNorm2d 层"""
    return LayerNorm2d(*args, **kwargs)
