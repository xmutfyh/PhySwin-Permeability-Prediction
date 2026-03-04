# utils/stat_utils.py
# 统一从标签文件中计算 k 和 T/V/P 的均值与标准差
# 默认标签文件路径: datasetone3/train/trainlabels.xlsx

import numpy as np
import pandas as pd

# 默认标签路径（统一用这个名字）
DEFAULT_LABEL_PATH = "datasetone3/train/trainlabels.xlsx"


def _load_labels(path: str = DEFAULT_LABEL_PATH) -> pd.DataFrame:
    """读取标签 Excel，确保至少包含 k 列。"""
    df = pd.read_excel(path)
    if "k" not in df.columns:
        raise ValueError(
            f"[stat_utils] 标签文件 {path} 中没有 'k' 列，请检查表头。"
        )
    return df


def compute_k_stats(path: str = DEFAULT_LABEL_PATH, eps: float = 1e-8):
    """
    计算 log(k+eps) 的 mean/std，和 pipeline 里的 LogTransform + StandardizeLabels 对齐。
    返回: (k_mean, k_std)
    """
    df = _load_labels(path)
    k_raw = df["k"].to_numpy().astype(float)
    k_log = np.log(k_raw + eps)

    k_mean = float(k_log.mean())
    k_std = float(k_log.std(ddof=0))  # 和 numpy 默认保持一致

    print(f"[stat_utils] k stats from {path}: mean={k_mean:.6f}, std={k_std:.6f}")
    return k_mean, k_std


def compute_tvp_stats(path: str = DEFAULT_LABEL_PATH):
    """
    计算 T/V/P 的 mean/std，用于 Phy 模型的物理量归一化。
    返回: (tvp_means, tvp_stds)，都是长度为 3 的 list，顺序为 [T, V, P]
    """
    df = _load_labels(path)
    required_cols = ["T", "V", "P"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"[stat_utils] 标签文件 {path} 中缺少列 '{c}'，当前列有: {list(df.columns)}"
            )

    tvp_means = [float(df[c].mean()) for c in required_cols]
    tvp_stds = [float(df[c].std(ddof=0)) for c in required_cols]

    print(
        "[stat_utils] TVP stats from {}:\n"
        "  means = {}\n"
        "  stds  = {}".format(path, tvp_means, tvp_stds)
    )
    return tvp_means, tvp_stds
