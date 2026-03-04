import math
import random
from typing import Sequence, Optional

import cv2
import numpy as np
from PIL import Image

from .build import PIPELINES_REGRESSION


def _to_numpy(img):
    if isinstance(img, Image.Image):
        return np.array(img)
    return img


def _clip01(img):
    return np.clip(img, 0.0, 1.0)


def _ensure_float01(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _to_uint8(img):
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


@PIPELINES_REGRESSION.register_module()
class GammaCorrection:
    """Gamma 校正.
    Args:
        gamma (float): 伽马值，>0。gamma>1 变暗，<1 变亮
        to_rgb (bool): 是否假设/输出为 RGB 顺序（仅用于兼容配置，不改变实现）
    """
    def __init__(self, gamma=2.2, to_rgb=True):
        self.gamma = float(gamma)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = _to_numpy(results['img'])
        img = _ensure_float01(img)
        img = np.power(img, 1.0 / max(self.gamma, 1e-6))
        results['img'] = _to_uint8(img)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(gamma={self.gamma}, to_rgb={self.to_rgb})'


@PIPELINES_REGRESSION.register_module()
class ContrastEnhancement:
    """线性对比度增强: out = alpha*img + beta
    Args:
        factor (float): alpha，>1 增强，<1 减弱
        beta (float): 亮度偏移（0-255 量纲）
    """
    def __init__(self, factor=1.2, beta=0.0):
        self.factor = float(factor)
        self.beta = float(beta)

    def __call__(self, results):
        img = _to_numpy(results['img'])
        out = cv2.convertScaleAbs(img, alpha=self.factor, beta=self.beta)
        results['img'] = out
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor}, beta={self.beta})'


@PIPELINES_REGRESSION.register_module()
class GaussianBlur:
    """高斯模糊.
    Args:
        kernel_size (int): 核大小(奇数)
        sigma (float): 标准差
    """
    def __init__(self, kernel_size=3, sigma=1.0):
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.sigma = float(sigma)

    def __call__(self, results):
        img = _to_numpy(results['img'])
        results['img'] = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})'


@PIPELINES_REGRESSION.register_module()
class AddGaussianNoise:
    """为图像添加高斯噪声.
    Args:
        mean (float): 均值（像素量纲，0-255）
        std (float): 标准差（像素量纲，0-255）
    """
    def __init__(self, mean=0.0, std=2.0):
        self.mean = float(mean)
        self.std = float(std)

    def __call__(self, results):
        img = _to_numpy(results['img']).astype(np.float32)
        noise = np.random.normal(self.mean, self.std, size=img.shape).astype(np.float32)
        noisy = img + noise
        if img.dtype == np.uint8:
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        results['img'] = noisy
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


@PIPELINES_REGRESSION.register_module()
class ColorJitter:
    """颜色抖动（亮度/对比度/饱和度）.
    注意: 灰度图时只影响亮度和对比度。
    Args:
        brightness (float): 范围 [max(0, 1-brightness), 1+brightness]
        contrast (float): 同上
        saturation (float): 同上（灰度无效）
        prob (float): 执行概率
    """
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, prob=1.0):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.prob = float(prob)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        img = _to_numpy(results['img'])
        img_f = _ensure_float01(img)

        # brightness
        if self.brightness > 0:
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            img_f = img_f * b

        # contrast
        if self.contrast > 0:
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = img_f.mean()
            img_f = (img_f - mean) * c + mean

        # saturation (only if 3 channels)
        if self.saturation > 0 and img_f.ndim == 3 and img_f.shape[2] == 3:
            gray = np.dot(img_f, [0.299, 0.587, 0.114])[..., None]
            s = 1.0 + random.uniform(-self.saturation, self.saturation)
            img_f = (img_f - gray) * s + gray

        results['img'] = _to_uint8(img_f)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(brightness={self.brightness}, '
                f'contrast={self.contrast}, saturation={self.saturation}, prob={self.prob})')


@PIPELINES_REGRESSION.register_module()
class RandomBrightnessContrastWrapper:
    """简化版随机亮度对比度（仿 albumentations 行为）.
    Args:
        brightness_limit (float): 亮度随机范围 [-limit, +limit]
        contrast_limit (float): 对比度随机范围 [-limit, +limit]
        prob (float): 概率
    """
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, prob=0.5):
        self.brightness_limit = float(brightness_limit)
        self.contrast_limit = float(contrast_limit)
        self.prob = float(prob)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        img = _to_numpy(results['img'])
        img_f = _ensure_float01(img)

        b = random.uniform(-self.brightness_limit, self.brightness_limit)
        c = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)

        mean = img_f.mean()
        img_f = (img_f - mean) * c + mean + b
        results['img'] = _to_uint8(img_f)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(brightness_limit={self.brightness_limit}, contrast_limit={self.contrast_limit}, prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class Normalize:
    """按均值/方差归一化.
    Args:
        mean (Sequence[float]): 每通道均值（0-255 量纲）
        std (Sequence[float]): 每通道方差（0-255 量纲）
        to_rgb (bool): 若为 BGR 输入是否转 RGB（仅兼容占位）
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float], to_rgb: bool = True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = _to_numpy(results['img']).astype(np.float32)
        if img.ndim == 2:
            img = img[..., None]
        if self.mean.size == 1 and img.shape[2] > 1:
            mean = np.full((img.shape[2],), float(self.mean[0]), dtype=np.float32)
            std = np.full((img.shape[2],), float(self.std[0]), dtype=np.float32)
        else:
            mean = self.mean
            std = self.std
        img = (img - mean) / np.maximum(std, 1e-6)
        if img.shape[2] == 1:
            img = img[..., 0]
        results['img'] = img
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean.tolist()}, std={self.std.tolist()}, to_rgb={self.to_rgb})'


@PIPELINES_REGRESSION.register_module()
class Lighting:
    """PCA-based lighting noise 的简化占位（默认不做任何事）.
    仅为满足 import; 若需要可扩展。
    """
    def __init__(self, alpha_std: float = 0.0):
        self.alpha_std = float(alpha_std)

    def __call__(self, results):
        # no-op
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha_std={self.alpha_std})'


@PIPELINES_REGRESSION.register_module()
class LogTransform:
    """对标签 k 做对数变换: k = log(offset + k)/log(base)
    Args:
        log_base (float): 对数底
        offset (float): 偏移，避免 log(0)
        key (str): 标签键名
    """
    def __init__(self, log_base=math.e, offset=0.0, key='k'):
        self.log_base = float(log_base)
        self.offset = float(offset)
        self.key = key

    def __call__(self, results):
        if self.key not in results:
            return results
        x = results[self.key]
        if hasattr(x, 'numpy'):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        x = np.log(np.maximum(x + self.offset, 1e-12)) / math.log(self.log_base)
        results[self.key] = x
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(log_base={self.log_base}, offset={self.offset}, key={self.key})'


@PIPELINES_REGRESSION.register_module()
class StandardizeLabels:
    """对标签 k 做标准化: (k-mean)/std
    Args:
        mean (float)
        std (float)
        key (str)
    """
    def __init__(self, mean, std, key='k'):
        self.mean = float(mean)
        self.std = float(std) if std != 0 else 1.0
        self.key = key

    def __call__(self, results):
        if self.key not in results:
            return results
        x = results[self.key]
        if hasattr(x, 'numpy'):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        x = (x - self.mean) / self.std
        results[self.key] = x
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, key={self.key})'


@PIPELINES_REGRESSION.register_module()
class SqrtTransform:
    """对标签 k 做平方根变换: sqrt(offset + k)"""
    def __init__(self, offset=0.0, key='k'):
        self.offset = float(offset)
        self.key = key

    def __call__(self, results):
        if self.key not in results:
            return results
        x = results[self.key]
        if hasattr(x, 'numpy'):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        x = np.sqrt(np.maximum(x + self.offset, 0.0))
        results[self.key] = x
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(offset={self.offset}, key={self.key})'


@PIPELINES_REGRESSION.register_module()
class BoxCoxTransform:
    """对标签 k 做 Box-Cox 变换.
    Args:
        lam (float): lambda 参数
        offset (float): 偏移，确保正数
        key (str)
    """
    def __init__(self, lam=0.0, offset=0.0, key='k'):
        self.lam = float(lam)
        self.offset = float(offset)
        self.key = key

    def __call__(self, results):
        if self.key not in results:
            return results
        x = results[self.key]
        if hasattr(x, 'numpy'):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        x = x + self.offset
        x = np.maximum(x, 1e-12)
        if abs(self.lam) < 1e-6:
            y = np.log(x)
        else:
            y = (np.power(x, self.lam) - 1.0) / self.lam
        results[self.key] = y
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(lam={self.lam}, offset={self.offset}, key={self.key})'


@PIPELINES_REGRESSION.register_module()
class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = float(prob)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        img = _to_numpy(results['img'])
        results['img'] = cv2.flip(img, 1)
        results['flip'] = True
        results['flip_direction'] = 'horizontal'
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = float(prob)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        img = _to_numpy(results['img'])
        results['img'] = cv2.flip(img, 0)
        results['flip'] = True
        results['flip_direction'] = 'vertical'
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


@PIPELINES_REGRESSION.register_module()
class RandomRotate:
    """随机旋转若干角度（简版）"""
    def __init__(self, angles: Sequence[int] = (-10, -5, 0, 5, 10), prob=0.5):
        self.angles = list(angles)
        self.prob = float(prob)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        img = _to_numpy(results['img'])
        h, w = img.shape[:2]
        angle = random.choice(self.angles)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        flags = cv2.INTER_LINEAR
        if img.ndim == 2:
            rotated = cv2.warpAffine(img, M, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT_101)
        else:
            rotated = cv2.warpAffine(img, M, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT_101)
        results['img'] = rotated
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(angles={self.angles}, prob={self.prob})'
