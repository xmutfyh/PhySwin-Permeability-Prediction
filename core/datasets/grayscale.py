import numpy as np
import cv2
from .build import PIPELINES_REGRESSION

@PIPELINES_REGRESSION.register_module()
class GrayscaleConversion:
    """Convert image to grayscale.
    Args:
        to_rgb (bool): if True, expand gray to 3-channel by replication (RGB-like).
    """
    def __init__(self, to_rgb=False):
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results['img']
        # PIL -> np
        if hasattr(img, 'mode'):
            img = np.array(img)
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            gray = img
        else:
            raise ValueError(f'Unsupported image shape for grayscale: {img.shape}')
        if self.to_rgb:
            gray = np.stack([gray, gray, gray], axis=-1)
        results['img'] = gray
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(to_rgb={self.to_rgb})'
