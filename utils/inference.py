import os
from copy import deepcopy
import numpy as np
import torch
import cv2
from core.datasets.compose import Compose
from utils.checkpoint import load_checkpoint
import matplotlib
from core.visualization import imshow_infos
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')


def init_model(model, data_cfg, device='cuda:0', mode='eval'):
    """Initialize a classifier/regressor from config."""
    if mode == 'train':
        if data_cfg.get('train').get('pretrained_flag') and data_cfg.get('train').get('pretrained_weights'):
            ckpt = data_cfg.get('train').get('pretrained_weights')
            print(f'Loading {os.path.basename(ckpt)}')
            load_checkpoint(model, ckpt, device, False)
    elif mode == 'eval':
        ckpt = data_cfg.get('test').get('ckpt')
        if not ckpt:
            raise ValueError('Please set data_cfg.test.ckpt before evaluation/inference.')
        print(f'Loading {os.path.basename(ckpt)}')
        model.eval()
        load_checkpoint(model, ckpt, device, False)

    model.to(device)
    return model


def inference_model(model, image, val_pipeline, label=None):
    """Inference single image.

    Args:
        model (nn.Module): The loaded model (on device).
        image (str/ndarray): Path or already-loaded image.
        val_pipeline (list[dict]): Compose pipeline config.
        label (float | tensor | None): Optional label passed to pipeline.
    Returns:
        dict: {'pred_value': float, 'image_path': str}
    """
    vp = deepcopy(val_pipeline)  # 避免原地修改
    if isinstance(image, str):
        if len(vp) == 0 or vp[0].get('type') != 'LoadImageFromFile':
            vp.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=image), img_prefix=None)
    else:
        if len(vp) > 0 and vp[0].get('type') == 'LoadImageFromFile':
            vp.pop(0)
        data = dict(img=image, filename=None)

    if label is not None:
        data['k'] = label

    pipeline = Compose(vp)
    data_out = pipeline(data)  # 只执行一次
    image_tensor = data_out['img'].unsqueeze(0)
    image_path = data_out['filename']
    device = next(model.parameters()).device

    with torch.no_grad():
        pred_value = model(image_tensor.to(device, dtype=torch.float32), return_loss=False).item()
        return {'pred_value': float(pred_value), 'image_path': image_path}


def inference_backbone(model, image, val_pipeline):
    """Extract backbone features for a single image."""
    vp = deepcopy(val_pipeline)
    if isinstance(image, str):
        if len(vp) == 0 or vp[0].get('type') != 'LoadImageFromFile':
            vp.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=image), img_prefix=None)
    else:
        if len(vp) > 0 and vp[0].get('type') == 'LoadImageFromFile':
            vp.pop(0)
        data = dict(img=image, filename=None)

    pipeline = Compose(vp)
    data_out = pipeline(data)
    image_tensor = data_out['img'].unsqueeze(0)
    device = next(model.parameters()).device

    with torch.no_grad():
        if hasattr(model, 'extract_feat'):
            feats = model.extract_feat(image_tensor.to(device, dtype=torch.float32))
        else:
            feats = model(image_tensor.to(device, dtype=torch.float32), return_loss=False)
    return feats
