# models/swin_transformer/tiny_224.py
# Swin-Tiny + PhyRegHead（融合 T/V/P）
import numpy as np
import pandas as pd

# ====== 训练集标签路径（按需修改）======
LABEL_PATH = "datasetone3/train/trainlabels.xlsx"

# 与 pipeline 保持一致的对数偏移（LogTransform 的 offset）
LOG_OFFSET = 1e-8


def _compute_k_stats_from_excel(label_path: str, column: str = "k", offset: float = LOG_OFFSET):
    """在本文件内计算 log(k+offset) 的均值/标准差（总体标准差，ddof=0）"""
    try:
        df = pd.read_excel(label_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read label excel: {label_path}. err={e}") from e

    if column not in df.columns:
        raise RuntimeError(f"Column {column!r} not found in {label_path}. Available: {list(df.columns)}")

    k = df[column].to_numpy(dtype=float)
    if np.isnan(k).any():
        raise RuntimeError("Found NaN in label column 'k'. Please clean your labels.")

    if np.min(k) + offset <= 0:
        raise RuntimeError(
            f"Found k <= {-offset} which makes log(k+offset) invalid. "
            f"Min k={np.min(k):.6f}, offset={offset}"
        )

    k_log = np.log(k + offset)
    k_mean = float(k_log.mean())
    k_std = float(k_log.std(ddof=0))
    if not np.isfinite(k_mean) or not np.isfinite(k_std) or k_std <= 0:
        raise RuntimeError(f"Invalid computed k stats: mean={k_mean}, std={k_std}")
    return k_mean, k_std


def _compute_tvp_stats_from_excel(label_path: str, columns=("T", "V", "P")):
    """计算 T/V/P 的均值和总体标准差（ddof=0）"""
    try:
        df = pd.read_excel(label_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read label excel: {label_path}. err={e}") from e

    cols = list(columns)
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f"Column {c!r} not found in {label_path}. Available: {list(df.columns)}")

    means, stds = [], []
    for c in cols:
        arr = df[c].to_numpy(dtype=float)
        if np.isnan(arr).any():
            raise RuntimeError(f"Found NaN in column {c!r}. Please clean your labels.")
        m = float(arr.mean())
        s = float(arr.std(ddof=0))
        if not np.isfinite(m) or not np.isfinite(s) or s <= 0:
            raise RuntimeError(f"Invalid stats for {c!r}: mean={m}, std={s}")
        means.append(m)
        stds.append(s)
    return means, stds


def _auto_k_stats(label_path: str):
    """优先使用 utils.stat_utils.compute_k_stats；若不可用则本地计算；失败则抛错"""
    try:
        from utils.stat_utils import compute_k_stats  # type: ignore
        k_mean, k_std = compute_k_stats(label_path)
        if not np.isfinite(k_mean) or not np.isfinite(k_std) or k_std <= 0:
            raise RuntimeError(f"compute_k_stats returned invalid values: mean={k_mean}, std={k_std}")
        return float(k_mean), float(k_std)
    except Exception as e1:
        try:
            return _compute_k_stats_from_excel(label_path)
        except Exception as e2:
            raise RuntimeError(
                f"Auto compute k stats failed. First error (stat_utils): {e1}; "
                f"Then local compute failed: {e2}"
            )


def _auto_tvp_stats(label_path: str):
    """优先使用 utils.stat_utils.compute_tvp_stats；若不可用则本地计算；失败则抛错"""
    try:
        from utils.stat_utils import compute_tvp_stats  # type: ignore
        means, stds = compute_tvp_stats(label_path)
        if len(means) != 3 or len(stds) != 3:
            raise RuntimeError(f"compute_tvp_stats must return 3 means and 3 stds, got {means}, {stds}")
        if not all(np.isfinite(means)) or not all(np.isfinite(stds)) or any(s <= 0 for s in stds):
            raise RuntimeError(f"compute_tvp_stats returned invalid values: means={means}, stds={stds}")
        return [float(x) for x in means], [float(x) for x in stds]
    except Exception as e1:
        try:
            return _compute_tvp_stats_from_excel(label_path)
        except Exception as e2:
            raise RuntimeError(
                f"Auto compute T/V/P stats failed. First error (stat_utils): {e1}; "
                f"Then local compute failed: {e2}"
            )


# ====== 计算并强制使用训练集统计值（失败直接抛错）======
k_mean, k_std = _auto_k_stats(LABEL_PATH)
tvp_means, tvp_stds = _auto_tvp_stats(LABEL_PATH)

print(f"[tiny_224] computed k stats from {LABEL_PATH}: mean={k_mean:.6f}, std={k_std:.6f}")
print(f"[tiny_224] computed tvp stats from {LABEL_PATH}: means={tvp_means}, stds={tvp_stds}")

# ====== 模型配置 ======
model_cfg = dict(
    backbone=dict(
        type='SwinTransformer',
        arch='tiny',
        img_size=224,
        in_channels=3,        # === 3 通道输入 ===
        drop_path_rate=0.05,
        drop_rate=0.0,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='PhyRegHead',
        in_channels=768,
        cond_dim=3,
        init_cfg=None,
        loss=dict(type='HuberLoss', loss_weight=1.0),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0),
    ],
)

####################################
#  数据预处理 / 数据增强 pipeline   #
####################################

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GrayscaleConversion', to_rgb=True),  # === 灰度复制到 3 通道 ===
    dict(type='GammaCorrection', gamma=2.5, to_rgb=False),
    dict(
        type='RandomBrightnessContrastWrapper',
        brightness_limit=0.3,
        contrast_limit=0.3,
        prob=0.5
    ),
    dict(type='GaussianBlur', kernel_size=3, sigma=1),
    dict(type='AddGaussianNoise', mean=0.0, std=2.0),

    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['T', 'V', 'P']),
    dict(type='ToTensor', keys=['k']),

    dict(
        type='StandardizeFields',
        keys=['T', 'V', 'P'],
        means=tvp_means,
        stds=tvp_stds
    ),

    dict(type='LogTransform', offset=LOG_OFFSET, log_base=np.e),
    dict(type='StandardizeLabels', mean=k_mean, std=k_std),

    dict(type='Collect', keys=['img', 'T', 'V', 'P', 'k', 'filename']),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GrayscaleConversion', to_rgb=True),  # === 3 通道 ===
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['T', 'V', 'P']),
    dict(type='ToTensor', keys=['k']),

    dict(
        type='StandardizeFields',
        keys=['T', 'V', 'P'],
        means=tvp_means,
        stds=tvp_stds
    ),

    dict(type='LogTransform', offset=LOG_OFFSET, log_base=np.e),
    dict(type='StandardizeLabels', mean=k_mean, std=k_std),
    dict(type='Collect', keys=['img', 'T', 'V', 'P', 'k', 'filename']),
]

test_pipeline = valid_pipeline

########################
#   数据 / 训练配置    #
########################

data_cfg = dict(
    batch_size=16,
    num_workers=2,

    train=dict(
        pretrained_flag=False,
        pretrained_weights='',
        freeze_flag=False,
        freeze_layers=('backbone',),
        epoches=100,
    ),

    valid=dict(
        batch_size=16,
        num_workers=2,
    ),

    test=dict(
        ckpt='',
        metrics=['rmse', 'r2', 'medare', 'mae'],  # === 四个指标 ===
        metric_options=dict(thrs=None, average_mode='none'),
    ),
)

############################
#   优化器 & 学习率策略    #
############################

optimizer_cfg = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=1e-4,
)

lr_config = dict(
    type='StepLrUpdater',
    step=[60, 80],
    gamma=0.1,
)
