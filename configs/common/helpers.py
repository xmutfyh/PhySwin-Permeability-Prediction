import collections.abc
import warnings
from itertools import repeat

import torch
from utils.version_utils import digit_version


def is_tracing() -> bool:
    """
    检查是否正在进行 TorchScript 跟踪。
    对于回归任务，我们需要确保在跟踪过程中不会丢失任何特征信息。
    """
    if digit_version(torch.__version__) >= digit_version('1.6.0'):
        on_trace = torch.jit.is_tracing()
        if isinstance(on_trace, bool):
            return on_trace
        else:
            return torch._C._is_tracing()
    else:
        warnings.warn(
            'torch.jit.is_tracing 仅在 v1.6.0 及之后版本支持。因此 is_tracing 自动返回 False。'
            '如果使用低版本进行跟踪，请手动设置 on_trace。',
            UserWarning
        )
        return False


# 从 PyTorch 内部代码中提取，用于将输入转换为 n 元组
def _ntuple(n):
    """
    返回一个函数，将输入转换为 n 元组。
    对于回归任务，我们可能需要调整数据的形状以适应不同模型输入。

    Args:
        n (int): 目标元组的长度。

    Returns:
        parse (function): 一个函数，将输入转换为 n 元组。
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


# 为特定元组长度创建快捷函数
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple  # 通用 n 元组转换器

