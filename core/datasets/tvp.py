from typing import List, Sequence
import numpy as np
import torch
from .build import PIPELINES_REGRESSION

@PIPELINES_REGRESSION.register_module()
class StandardizeFields(object):
    """Standardize multiple scalar fields, e.g., T/V/P.
    Args:
        keys (list[str]): fields to standardize
        means (list[float]): per-field mean
        stds (list[float]): per-field std
        eps (float): small value to avoid div-by-zero
        clip (float|None): optional absolute clip value (apply after standardize)
    """
    def __init__(self,
                 keys: Sequence[str],
                 means: Sequence[float],
                 stds: Sequence[float],
                 eps: float = 1e-6,
                 clip: float = None):
        assert len(keys) == len(means) == len(stds) and len(keys) > 0
        self.keys = list(keys)
        self.means = [float(m) for m in means]
        self.stds = [float(s) for s in stds]
        self.eps = float(eps)
        self.clip = None if clip is None else float(clip)

    def _standardize_np(self, arr: np.ndarray, mean: float, std: float) -> np.ndarray:
        x = arr.astype(np.float32)
        x = (x - mean) / max(std, self.eps)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return x.astype(np.float32)

    def _standardize_torch(self, ten: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        x = ten.to(dtype=torch.float32)
        x = (x - mean) / max(std, self.eps)
        if self.clip is not None:
            x = torch.clamp(x, -self.clip, self.clip)
        return x

    def __call__(self, results: dict):
        for i, k in enumerate(self.keys):
            if k not in results:
                continue
            mean, std = self.means[i], self.stds[i]
            v = results[k]
            if torch.is_tensor(v):
                results[k] = self._standardize_torch(v, mean, std)
            else:
                results[k] = self._standardize_np(np.asarray(v), mean, std)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'
