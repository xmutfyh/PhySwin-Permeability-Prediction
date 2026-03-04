from collections.abc import Sequence
import numpy as np
import torch
from PIL import Image
from .build import PIPELINES_REGRESSION

def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return torch.tensor(data)
    elif isinstance(data, (int, np.integer)):
        return torch.tensor([int(data)], dtype=torch.long)
    elif isinstance(data, (float, np.floating)):
        return torch.tensor([float(data)], dtype=torch.float)
    else:
        raise TypeError(f'Cannot convert type {type(data)} to tensor')

@PIPELINES_REGRESSION.register_module()
class ToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            if key not in results:
                continue
            results[key] = to_tensor(results[key]).float()
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'

@PIPELINES_REGRESSION.register_module()
class ImageToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if isinstance(img, Image.Image):
                img = np.array(img)
            if img.ndim == 2:
                img = img[..., None]
            # HWC -> CHW
            img = img.transpose(2, 0, 1).astype(np.float32)
            results[key] = to_tensor(img)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'

@PIPELINES_REGRESSION.register_module()
class Transpose:
    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys}, order={self.order})'

@PIPELINES_REGRESSION.register_module()
class ToPIL:
    def __call__(self, results):
        results['img'] = Image.fromarray(results['img'])
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'

@PIPELINES_REGRESSION.register_module()
class ToNumpy:
    def __call__(self, results):
        results['img'] = np.array(results['img'], dtype=np.float32)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'

@PIPELINES_REGRESSION.register_module()
class Collect:
    """Collect specified keys; keep simple meta; add a 'meta' dict for convenience."""
    def __init__(self, keys, meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'flip', 'flip_direction')):
        self.keys = list(keys)
        self.meta_keys = list(meta_keys)

    def __call__(self, results):
        out = {}
        for k in self.keys:
            if k in results:
                out[k] = results[k]
        meta = {k: results[k] for k in self.meta_keys if k in results}
        if meta:
            out['meta'] = meta
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'
