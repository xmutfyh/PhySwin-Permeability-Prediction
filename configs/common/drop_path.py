import torch
import torch.nn as nn


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob=0.1, adjust_prob=False, warmup_steps=1000):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.adjust_prob = adjust_prob
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def forward(self, x):
        if self.adjust_prob and self.training:
            # 动态调整drop_prob
            adjusted_prob = self.drop_prob * (1 - self.current_step / self.warmup_steps)
            adjusted_prob = max(0.0, adjusted_prob)
            self.current_step += 1
        else:
            adjusted_prob = self.drop_prob
        return drop_path(x, adjusted_prob, self.training)

    def reset(self):
        """Reset the current step counter, useful for new training phases."""
        self.current_step = 0
