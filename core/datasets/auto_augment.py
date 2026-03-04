# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from math import ceil
from numbers import Number
from typing import Sequence

from .geometric import cutout, imrotate, imshear, imtranslate
from .photometric import (adjust_brightness, adjust_color, adjust_contrast, adjust_sharpness,
                          auto_contrast, imequalize, iminvert, posterize, solarize)
from .compose import Compose
from .build import PIPELINES_REGRESSION

# optional albumentations
try:
    from albumentations import RandomBrightnessContrast as AlbRandomBrightnessContrast
    _HAS_ALB = True
except Exception:
    _HAS_ALB = False

_HPARAMS_DEFAULT_REGRESSION = dict(pad_val=128, regression_param=0.5)


def random_negative(value, random_negative_prob):
    return -value if np.random.rand() < random_negative_prob else value


def merge_hparams(policy: dict, hparams: dict):
    op = PIPELINES_REGRESSION.get(policy['type'])
    assert op is not None, f'Invalid policy type "{policy["type"]}".'
    for key, value in hparams.items():
        if policy.get(key, None) is not None:
            continue
        if key in inspect.getfullargspec(op.__init__).args:
            policy[key] = value
    return policy


@PIPELINES_REGRESSION.register_module()
class RandomBrightnessContrastWrapper(object):
    """随机调整亮度/对比度；若安装了 albumentations 则用其实现，否则使用简易回退版本。"""
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, prob=0.5):
        assert 0 <= prob <= 1.0
        self.prob = prob
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        if _HAS_ALB:
            self._alb = AlbRandomBrightnessContrast(brightness_limit=brightness_limit,
                                                    contrast_limit=contrast_limit,
                                                    p=prob)
        else:
            self._alb = None

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if _HAS_ALB:
                augmented = self._alb(image=img)
                results[key] = augmented['image'].astype(img.dtype)
            else:
                if np.random.rand() > self.prob:
                    continue
                b_lim = self.brightness_limit
                c_lim = self.contrast_limit
                bmin, bmax = (b_lim if isinstance(b_lim, tuple) else (-b_lim, b_lim))
                cmin, cmax = (c_lim if isinstance(c_lim, tuple) else (-c_lim, c_lim))
                b, c = 1.0 + np.random.uniform(bmin, bmax), 1.0 + np.random.uniform(cmin, cmax)
                pil = Image.fromarray(img)
                pil = ImageEnhance.Brightness(pil).enhance(b)
                pil = ImageEnhance.Contrast(pil).enhance(c)
                results[key] = np.array(pil).astype(img.dtype)
        return results


@PIPELINES_REGRESSION.register_module()
class AutoAugment(object):
    def __init__(self, policies, hparams=_HPARAMS_DEFAULT_REGRESSION):
        assert isinstance(policies, list) and len(policies) > 0
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment

        self.hparams = hparams
        policies = copy.deepcopy(policies)
        self.policies = []
        for sub in policies:
            merged_sub = [merge_hparams(policy, hparams) for policy in sub]
            self.policies.append(merged_sub)
        self.sub_policy = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        sub_policy = random.choice(self.sub_policy)
        return sub_policy(results)

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies})'


@PIPELINES_REGRESSION.register_module()
class RandAugment(object):
    def __init__(self,
                 policies,
                 num_policies,
                 magnitude_level,
                 magnitude_std=0.,
                 total_level=30,
                 hparams=_HPARAMS_DEFAULT_REGRESSION):
        assert isinstance(num_policies, int)
        assert isinstance(magnitude_level, (int, float))
        assert isinstance(total_level, (int, float))
        assert isinstance(policies, list) and len(policies) > 0
        assert isinstance(magnitude_std, (Number, str))
        if isinstance(magnitude_std, str):
            assert magnitude_std == 'inf'
        assert num_policies > 0
        assert magnitude_level >= 0
        assert total_level > 0

        self.num_policies = num_policies
        self.magnitude_level = magnitude_level
        self.magnitude_std = magnitude_std
        self.total_level = total_level
        self.hparams = hparams
        policies = copy.deepcopy(policies)
        self._check_policies(policies)
        self.policies = [merge_hparams(policy, hparams) for policy in policies]

    def _check_policies(self, policies):
        for policy in policies:
            assert isinstance(policy, dict) and 'type' in policy
            magnitude_key = policy.get('magnitude_key', None)
            if magnitude_key is not None:
                assert 'magnitude_range' in policy
                magnitude_range = policy['magnitude_range']
                assert (isinstance(magnitude_range, Sequence)
                        and len(magnitude_range) == 2)

    def _process_policies(self, policies):
        processed_policies = []
        for policy in policies:
            processed_policy = copy.deepcopy(policy)
            magnitude_key = processed_policy.pop('magnitude_key', None)
            if magnitude_key is not None:
                magnitude = self.magnitude_level
                if self.magnitude_std == 'inf':
                    magnitude = random.uniform(0, magnitude)
                elif self.magnitude_std > 0:
                    magnitude = random.gauss(magnitude, self.magnitude_std)
                    magnitude = min(self.total_level, max(0, magnitude))
                val1, val2 = processed_policy.pop('magnitude_range')
                magnitude = (magnitude / self.total_level) * (val2 - val1) + val1
                processed_policy.update({magnitude_key: magnitude})
            processed_policies.append(processed_policy)
        return processed_policies

    def __call__(self, results):
        if self.num_policies == 0:
            return results
        sub_policy = random.choices(self.policies, k=self.num_policies)
        sub_policy = self._process_policies(sub_policy)
        sub_policy = Compose(sub_policy)
        return sub_policy(results)

    def __repr__(self):
        return (f'{self.__class__.__name__}(policies={self.policies}, '
                f'num_policies={self.num_policies}, magnitude_level={self.magnitude_level}, '
                f'total_level={self.total_level})')


@PIPELINES_REGRESSION.register_module()
class Shear(object):
    def __init__(self,
                 magnitude,
                 pad_val=128,
                 prob=0.5,
                 direction='horizontal',
                 random_negative_prob=0.5,
                 interpolation='bicubic'):
        assert isinstance(magnitude, (int, float))
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3 and all(isinstance(i, int) for i in pad_val)
        else:
            raise TypeError('pad_val must be int or tuple with 3 elements.')
        assert 0 <= prob <= 1.0
        assert direction in ('horizontal', 'vertical')
        assert 0 <= random_negative_prob <= 1.0

        self.magnitude = magnitude
        self.pad_val = tuple(pad_val)
        self.prob = prob
        self.direction = direction
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sheared = imshear(
                img, magnitude, direction=self.direction,
                border_value=self.pad_val, interpolation=self.interpolation)
            results[key] = img_sheared.astype(img.dtype)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(magnitude={self.magnitude}, pad_val={self.pad_val}, '
                f'prob={self.prob}, direction={self.direction}, '
                f'random_negative_prob={self.random_negative_prob}, interpolation={self.interpolation})')


@PIPELINES_REGRESSION.register_module()
class Translate(object):
    def __init__(self,
                 magnitude,
                 pad_val=128,
                 prob=0.5,
                 direction='horizontal',
                 random_negative_prob=0.5,
                 interpolation='nearest'):
        assert isinstance(magnitude, (int, float))
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3 and all(isinstance(i, int) for i in pad_val)
        else:
            raise TypeError('pad_val must be int or tuple with 3 elements.')
        assert 0 <= prob <= 1.0
        assert direction in ('horizontal', 'vertical')
        assert 0 <= random_negative_prob <= 1.0

        self.magnitude = magnitude
        self.pad_val = tuple(pad_val)
        self.prob = prob
        self.direction = direction
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            height, width = img.shape[:2]
            offset = magnitude * (width if self.direction == 'horizontal' else height)
            img_translated = imtranslate(
                img, offset, direction=self.direction,
                border_value=self.pad_val, interpolation=self.interpolation)
            results[key] = img_translated.astype(img.dtype)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(magnitude={self.magnitude}, pad_val={self.pad_val}, '
                f'prob={self.prob}, direction={self.direction}, '
                f'random_negative_prob={self.random_negative_prob}, interpolation={self.interpolation})')


@PIPELINES_REGRESSION.register_module()
class Rotate(object):
    def __init__(self,
                 angle,
                 center=None,
                 scale=1.0,
                 pad_val=128,
                 prob=0.5,
                 random_negative_prob=0.5,
                 interpolation='nearest'):
        assert isinstance(angle, float)
        if isinstance(center, tuple):
            assert len(center) == 2
        else:
            assert center is None
        assert isinstance(scale, float)
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3 and all(isinstance(i, int) for i in pad_val)
        else:
            raise TypeError('pad_val must be int or tuple with 3 elements.')
        assert 0 <= prob <= 1.0
        assert 0 <= random_negative_prob <= 1.0

        self.angle = angle
        self.center = center
        self.scale = scale
        self.pad_val = tuple(pad_val)
        self.prob = prob
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        angle = random_negative(self.angle, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_rotated = imrotate(
                img, angle, center=self.center, scale=self.scale,
                border_value=self.pad_val, interpolation=self.interpolation)
            results[key] = img_rotated.astype(img.dtype)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(angle={self.angle}, center={self.center}, '
                f'scale={self.scale}, pad_val={self.pad_val}, prob={self.prob}, '
                f'random_negative_prob={self.random_negative_prob}, interpolation={self.interpolation})')


@PIPELINES_REGRESSION.register_module()
class AutoContrast(object):
    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_contrasted = auto_contrast(img)
            results[key] = img_contrasted.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class Invert(object):
    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_inverted = iminvert(img)
            results[key] = img_inverted.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class Equalize(object):
    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_equalized = imequalize(img)
            results[key] = img_equalized.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class Solarize(object):
    def __init__(self, thr, prob=0.5):
        assert isinstance(thr, (int, float))
        assert 0 <= prob <= 1.0
        self.thr = thr
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_solarized = solarize(img, thr=self.thr)
            results[key] = img_solarized.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(thr={self.thr}, prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class SolarizeAdd(object):
    def __init__(self, magnitude, thr=128, prob=0.5):
        assert isinstance(magnitude, (int, float))
        assert isinstance(thr, (int, float))
        assert 0 <= prob <= 1.0
        self.magnitude = magnitude
        self.thr = thr
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_solarized = np.where(img < self.thr,
                                     np.minimum(img + self.magnitude, 255),
                                     img)
            results[key] = img_solarized.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(magnitude={self.magnitude}, thr={self.thr}, prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class Posterize(object):
    def __init__(self, bits, prob=0.5):
        assert bits <= 8
        assert 0 <= prob <= 1.0
        self.bits = ceil(bits)
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_posterized = posterize(img, bits=self.bits)
            results[key] = img_posterized.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(bits={self.bits}, prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class Contrast(object):
    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float))
        assert 0 <= prob <= 1.0
        assert 0 <= random_negative_prob <= 1.0
        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_contrasted = adjust_contrast(img, factor=1 + magnitude)
            results[key] = img_contrasted.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(magnitude={self.magnitude}, prob={self.prob}, random_negative_prob={self.random_negative_prob})'


@PIPELINES_REGRESSION.register_module()
class ColorTransform(object):
    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float))
        assert 0 <= prob <= 1.0
        assert 0 <= random_negative_prob <= 1.0
        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_color_adjusted = adjust_color(img, alpha=1 + magnitude)
            results[key] = img_color_adjusted.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(magnitude={self.magnitude}, prob={self.prob}, random_negative_prob={self.random_negative_prob})'


@PIPELINES_REGRESSION.register_module()
class Brightness(object):
    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float))
        assert 0 <= prob <= 1.0
        assert 0 <= random_negative_prob <= 1.0
        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_brightened = adjust_brightness(img, factor=1 + magnitude)
            results[key] = img_brightened.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(magnitude={self.magnitude}, prob={self.prob}, random_negative_prob={self.random_negative_prob})'


@PIPELINES_REGRESSION.register_module()
class Sharpness(object):
    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float))
        assert 0 <= prob <= 1.0
        assert 0 <= random_negative_prob <= 1.0
        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sharpened = adjust_sharpness(img, factor=1 + magnitude)
            results[key] = img_sharpened.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(magnitude={self.magnitude}, prob={self.prob}, random_negative_prob={self.random_negative_prob})'


@PIPELINES_REGRESSION.register_module()
class Cutout(object):
    def __init__(self, shape, pad_val=128, prob=0.5):
        if isinstance(shape, float):
            shape = int(shape)
        elif isinstance(shape, tuple):
            shape = tuple(int(i) for i in shape)
        elif not isinstance(shape, int):
            raise TypeError('shape must be int/float/tuple')
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3
        assert 0 <= prob <= 1.0

        self.shape = shape
        self.pad_val = tuple(pad_val)
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_cut = cutout(img, self.shape, pad_val=self.pad_val)
            results[key] = img_cut.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.shape}, pad_val={self.pad_val}, prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_flipped = np.flipud(img)
            results[key] = img_flipped.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_flipped = np.fliplr(img)
            results[key] = img_flipped.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class RandomRotate(object):
    """按角度范围随机旋转（简单版，PIL/np/cv2 皆可）。"""
    def __init__(self, degrees):
        if isinstance(degrees, int) or isinstance(degrees, float):
            degrees = (-float(degrees), float(degrees))
        elif isinstance(degrees, tuple):
            degrees = (float(degrees[0]), float(degrees[1]))
        else:
            raise TypeError('degrees must be int/float/tuple')
        self.degrees = degrees

    def __call__(self, results):
        angle = float(np.random.uniform(self.degrees[0], self.degrees[1]))
        for key in results.get('img_fields', ['img']):
            img = results[key]
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            results[key] = img_rotated.astype(img.dtype)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(degrees={self.degrees})'
