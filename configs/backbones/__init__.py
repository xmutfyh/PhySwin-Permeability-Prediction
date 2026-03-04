__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d'
    , 'DenseNet',
    'SwinTransformer', 'ResNetTransformer',
     'CNNFairBackbone' ,   # ★ 新增 SimpleCNN
]

from configs.backbones.densenet import DenseNet
from configs.backbones.resnet import ResNet, ResNetV1c, ResNetV1d
from configs.backbones.swin_transformer import SwinTransformer
from configs.backbones.resnettransformer import ResNetTransformer


from .cnn_fair_backbone import CNNFairBackbone
