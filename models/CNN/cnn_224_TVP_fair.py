# models/CNN/cnn_224_TVP_fair.py
# CNNFairBackbone + GAP + PhyRegHead (TVP via head late-fusion)

import numpy as np
import pandas as pd

# ====== 训练集标签路径（按需修改）======
LABEL_PATH = "datasetone3/train/trainlabels.xlsx"

# 与 pipeline 保持一致的对数偏移
LOG_OFFSET = 1e-8


def _compute_stats_from_excel(label_path: str, offset: float = LOG_OFFSET):
    df = pd.read_excel(label_path)

    for c in ("k", "T", "V", "P"):
        if c not in df.columns:
            raise RuntimeError(f"Column {c!r} not found in {label_path}. Available: {list(df.columns)}")

    k = df["k"].to_numpy(dtype=float)
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

    means, stds = [], []
    for c in ("T", "V", "P"):
        arr = df[c].to_numpy(dtype=float)
        if np.isnan(arr).any():
            raise RuntimeError(f"Found NaN in column {c!r}. Please clean your labels.")
        m = float(arr.mean())
        s = float(arr.std(ddof=0))
        if not np.isfinite(m) or not np.isfinite(s) or s <= 0:
            raise RuntimeError(f"Invalid stats for {c!r}: mean={m}, std={s}")
        means.append(m)
        stds.append(s)

    return k_mean, k_std, means, stds


K_MEAN, K_STD, TVP_MEANS, TVP_STDS = _compute_stats_from_excel(LABEL_PATH)
print(f"[cnn_224_TVP_fair] K_MEAN={K_MEAN:.6f}, K_STD={K_STD:.6f}, LOG_OFFSET={LOG_OFFSET:g}")

# ===========================
# Model
# ===========================
model_cfg = dict(
    backbone=dict(
        type='CNNFairBackbone',
        in_channels=3,
        base_channels=64,
        out_channels=768,
        num_blocks=(2, 2, 6, 2),

        # ✅ 关键：TVP 交给 PhyRegHead 融合，所以 backbone 这里关掉
        use_tvp=False,
        tvp_dim=3,
        tvp_hidden=128,

        init_cfg=None,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='PhyRegHead',
        in_channels=768,
        cond_dim=3,  # ✅ 使用 T/V/P
        hidden_dims=(512, 128),
        dropout=0.1,
        loss=dict(type='HuberLoss', loss_weight=1.0),
        init_cfg=None,
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0),
    ],
)

###########################################
#             训练集 pipeline
###########################################
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GrayscaleConversion', to_rgb=True),

    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),

    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['T', 'V', 'P']),
    dict(type='ToTensor', keys=['k']),

    dict(
        type='StandardizeFields',
        keys=['T', 'V', 'P'],
        means=TVP_MEANS,
        stds=TVP_STDS
    ),

    dict(type='LogTransform', offset=LOG_OFFSET, log_base=np.e),
    dict(type='StandardizeLabels', mean=K_MEAN, std=K_STD),

    dict(type='Collect', keys=['img', 'T', 'V', 'P', 'k', 'filename']),
]

###########################################
#             验证集 pipeline
###########################################
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GrayscaleConversion', to_rgb=True),

    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),

    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['T', 'V', 'P']),
    dict(type='ToTensor', keys=['k']),

    dict(
        type='StandardizeFields',
        keys=['T', 'V', 'P'],
        means=TVP_MEANS,
        stds=TVP_STDS
    ),

    dict(type='LogTransform', offset=LOG_OFFSET, log_base=np.e),
    dict(type='StandardizeLabels', mean=K_MEAN, std=K_STD),

    dict(type='Collect', keys=['img', 'T', 'V', 'P', 'k', 'filename']),
]

###########################################
#             测试集 pipeline
###########################################
test_pipeline = valid_pipeline.copy()

###########################################
#               训练配置（train.py 需要）
###########################################
data_cfg = dict(
    batch_size=8,
    num_workers=2,
    train=dict(
        pretrained_flag=False,
        pretrained_weights='',
        freeze_flag=False,
        freeze_layers=('backbone',),
        epoches=100,
    ),
    valid=dict(
        batch_size=8,
        num_workers=2,
    ),
    test=dict(
        ckpt='',
        metrics=['rmse', 'r2'],
        metric_options=dict(thrs=None, average_mode='none'),
    ),
)

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
