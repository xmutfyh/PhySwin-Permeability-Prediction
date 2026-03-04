import argparse
import os
import sys
sys.path.insert(0, os.getcwd())

from utils.train_utils import file2dict
from utils.flops_counter import get_model_complexity_info
from models.build import BuildNet


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image spatial size: H W or single value (assumed square)')
    parser.add_argument(
        '--channels',
        type=int,
        default=None,
        help='override input channels (default: auto-detect from model, fallback=3)'
    )
    args = parser.parse_args()
    return args


def infer_in_channels(model):
    """Best-effort to infer expected input channels from backbone."""
    # common patterns
    try:
        # SwinTransformer: backbone.patch_embed.projection.in_channels
        pe = getattr(model.backbone, 'patch_embed', None)
        if pe is not None:
            proj = getattr(pe, 'projection', None)
            if proj is not None and hasattr(proj, 'in_channels'):
                return int(proj.in_channels)
    except Exception:
        pass
    try:
        # ResNet-like: backbone.conv1.in_channels
        conv1 = getattr(model.backbone, 'conv1', None)
        if conv1 is not None and hasattr(conv1, 'in_channels'):
            return int(conv1.in_channels)
    except Exception:
        pass
    # fallback
    return 3


def main():
    args = parse_args()

    # parse H,W
    if len(args.shape) == 1:
        H = W = int(args.shape[0])
    elif len(args.shape) == 2:
        H, W = int(args.shape[0]), int(args.shape[1])
    else:
        raise ValueError('invalid --shape; expect one value (square) or two values H W')

    # load config/model
    model_cfg, train_pipeline, val_pipeline, test_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    model = BuildNet(model_cfg)
    model.eval()

    # infer channels or use override
    if args.channels is not None:
        C = int(args.channels)
    else:
        C = infer_in_channels(model)

    input_shape = (C, H, W)

    # use extract_feat for FLOPs
    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            f'FLOPs counter is not supported with {model.__class__.__name__}')

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
