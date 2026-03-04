import torch
import torch.nn as nn


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    with torch.no_grad():
        # 获取卷积层的权重和偏差
        conv_w = conv.weight.clone()
        conv_b = conv.bias.clone() if conv.bias is not None else torch.zeros_like(bn.running_mean)

        # 计算批归一化的缩放因子
        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)

        # 调整卷积层的权重和偏差
        conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
        conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)

    return conv

def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Recursively fuse conv and bn in a module.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, (nn.Conv2d, nn.Conv1d)):
            last_conv = child
            last_conv_name = name
        elif isinstance(child, nn.Linear):
            last_conv = None
            last_conv_name = None
        elif isinstance(child, nn.Module):
            # 对于其他子模块递归调用
            fuse_conv_bn(child)

    return module
