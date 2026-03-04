import os
import os.path as osp
import numpy as np
from core.datasets.io import imfrombytes
from .build import PIPELINES_REGRESSION


@PIPELINES_REGRESSION.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        
    def get(self,filepath):
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def __call__(self, results):
        # print("Results before loading image:", results)  # 添加调试信息
        filename = results['img_info']['filename']
        if results['img_prefix'] is not None:
            filename = os.path.join(results['img_prefix'], filename)

        img_bytes = self.get(filename)
        img = imfrombytes(img_bytes, flag=self.color_type)
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # print(f"After LoadImageFromFile: {results.keys()}")  # 调试打印
        return results

