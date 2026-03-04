from collections.abc import Sequence
import copy

from .build import build_from_cfg, PIPELINES_REGRESSION

@PIPELINES_REGRESSION.register_module()
class Compose(object):
    """Compose a data pipeline for regression tasks with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence), \
            f'transforms should be a sequence, but got {type(transforms)}'
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform_cfg = copy.deepcopy(transform)
                transform = build_from_cfg(transform_cfg, PIPELINES_REGRESSION)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            # print(f"Before transform {t.__class__.__name__}: {data.keys()}")  # 调试信息
            data = t(data)
            # print(f"After transform {t.__class__.__name__}: {data.keys()}")  # 调试信息
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = f'{self.__class__.__name__}('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string
