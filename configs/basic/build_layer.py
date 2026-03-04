import inspect
import copy
import torch.nn as nn
import torch.nn.functional as F
from .activations import *
from .convolution import *
from .normalization import *
from .padding import *
from .drop import *
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

# 维持现有的列表结构
CONV_LAYERS = ['Conv1d', 'Conv2d', 'Conv3d', 'Conv', 'Conv2dAdaptivePadding']
NORM_LAYERS = ['BN', 'BN1d', 'BN2d', 'BN3d', 'SyncBN', 'GN', 'LN', 'IN', 'IN1d', 'IN2d', 'IN3d', 'LN2d']
PADDING_LAYERS = ['ZeroPad2d', 'ReflectionPad2d', 'ReplicationPad2d']

LAYER_TYPE_TO_CLASS = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Conv': nn.Conv2d,
    'Conv2dAdaptivePadding': nn.Conv2d,  # 假设为 Conv2d
    'ZeroPad2d': nn.ZeroPad2d,
    'ReflectionPad2d': nn.ReflectionPad2d,
    'ReplicationPad2d': nn.ReplicationPad2d,
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'IN': nn.InstanceNorm2d,
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
    'LN2d': nn.LayerNorm,
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'Softmax': nn.Softmax,
    'LogSoftmax': nn.LogSoftmax,
    'Identity': nn.Identity,  # 添加 Identity 激活函数
}

def build_conv_layer(cfg, *args, **kwargs):
    """构建卷积层。

    Args:
        cfg (dict): 卷积层的配置，应包含:
            - type (str): 卷积层类型。
            - 其他参数: 用于实例化卷积层的参数。
        args: 传递给卷积层 `__init__` 方法的额外参数。
        kwargs: 传递给卷积层 `__init__` 方法的额外关键字参数。

    Returns:
        nn.Module: 创建的卷积层。
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg 必须是一个 dict')
        if 'type' not in cfg:
            raise KeyError('cfg 必须包含键 "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'未识别的卷积层类型 {layer_type}')

    conv_layer_class = LAYER_TYPE_TO_CLASS.get(layer_type)
    if conv_layer_class is None:
        raise ValueError(f"卷积层类型 '{layer_type}' 不受支持。")

    return conv_layer_class(*args, **kwargs, **cfg_)

def infer_abbr(class_type):
    """根据类名推断缩写。

    Args:
        class_type (type): 归一化层类型。

    Returns:
        str: 推断出的缩写。
    """
    if not inspect.isclass(class_type):
        raise TypeError(f'class_type 必须是一个类型，但得到 {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm'

def build_norm_layer(cfg, num_features, postfix=''):
    """构建归一化层。

    Args:
        cfg (dict): 归一化层的配置，应包含:
            - type (str): 归一化层类型。
            - 其他参数: 用于实例化归一化层的参数。
            - requires_grad (bool, optional): 是否需要梯度。
        num_features (int): 输入特征数量。
        postfix (int | str): 用于命名层的后缀。

    Returns:
        tuple[str, nn.Module]: 层名称和创建的归一化层。
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg 必须是一个 dict')
    if 'type' not in cfg:
        raise KeyError('cfg 必须包含键 "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'未识别的归一化层类型 {layer_type}')

    norm_layer_class = LAYER_TYPE_TO_CLASS.get(layer_type)
    abbr = infer_abbr(norm_layer_class)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer_class(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer_class(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def build_activation_layer(cfg):
    """构建激活层。

    Args:
        cfg (dict): 激活层的配置，应包含:
            - type (str): 激活层类型。
            - 其他参数: 用于实例化激活层的参数。

    Returns:
        nn.Module: 创建的激活层。
    """
    cfg_ = copy.deepcopy(cfg)
    return eval(cfg_.pop('type'))(**cfg_)

def build_padding_layer(cfg, *args, **kwargs):
    """构建填充层。

    Args:
        cfg (dict): 填充层的配置，应包含:
            - type (str): 填充层类型。
            - 其他参数: 用于实例化填充层的参数。

    Returns:
        nn.Module: 创建的填充层。
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg 必须是一个 dict')
    if 'type' not in cfg:
        raise KeyError('cfg 必须包含键 "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDING_LAYERS:
        raise KeyError(f'未识别的填充层类型 {padding_type}')

    padding_layer_class = LAYER_TYPE_TO_CLASS.get(padding_type)
    if padding_layer_class is None:
        raise ValueError(f"填充层类型 '{padding_type}' 不受支持。")

    return padding_layer_class(*args, **kwargs, **cfg_)

def build_dropout(cfg):
    """构建 Dropout 层。

    Args:
        cfg (dict): Dropout 层的配置，应包含:
            - type (str): Dropout 层类型。
            - 其他参数: 用于实例化 Dropout 层的参数。

    Returns:
        nn.Module: 创建的 Dropout 层。
    """
    cfg_ = cfg.copy()
    return eval(cfg_.pop('type'))(**cfg_)
