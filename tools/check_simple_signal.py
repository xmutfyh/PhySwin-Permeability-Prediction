# tools/check_simple_signal.py
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 保证能找到 utils / configs 等模块（和 train.py 做法一致）
sys.path.insert(0, os.getcwd())

from utils.dataloader import MyDataset, collate
from utils.train_utils import file2dict


def load_labels(file_path: str) -> pd.DataFrame:
    """读取 Excel 标签表（和 train.py 保持一致）"""
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"[ERROR] 加载标签失败: {file_path}, {e}")
        sys.exit(1)


def fit_linear_and_r2(X: np.ndarray, y: np.ndarray):
    """
    用最简单的线性回归 (带截距) 拟合:
        y ≈ w0 + w1 * x1 + w2 * x2 + ...
    然后计算 RMSE 和 R²
    """
    # 设计矩阵，加一列常数 1 作为截距
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  # (N, d+1)

    # 最小二乘解
    w, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ w

    # 计算 RMSE 和 R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return rmse, r2, w


def main():
    parser = argparse.ArgumentParser(
        description="用简单特征(均值/方差) vs k 做一个 R² 检查，看看图像里有没有 signal"
    )
    parser.add_argument(
        "config",
        help="配置文件路径，如 configs/tiny_224_noTVP.py（和 train.py 用的一样）",
    )
    args = parser.parse_args()

    # 1. 读 config，拿到 train_pipeline 和 data_cfg（和 train.py 完全一致）
    model_cfg, train_pipeline, valid_pipeline, test_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(
        args.config
    )

    print("[INFO] 使用配置文件:", args.config)
    print("[INFO] batch_size =", data_cfg.get("batch_size", 8),
          "num_workers =", data_cfg.get("num_workers", 2))

    # 2. 读 train 标签 Excel（路径和 train.py 保持一致）
    train_labels_path = "datasetone3/train/trainlabels.xlsx"
    if not os.path.exists(train_labels_path):
        print(f"[ERROR] 找不到 train labels 文件: {train_labels_path}")
        sys.exit(1)

    train_labels = load_labels(train_labels_path)

    # 3. 构造 Dataset 和 DataLoader（完全沿用 MyDataset + train_pipeline）
    train_dataset = MyDataset(
        train_labels,
        image_dir="datasetone3/train/",
        cfg=train_pipeline,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,  # 检查统计，不需要 shuffle
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    print("[INFO] 训练样本数:", len(train_dataset))

    all_feats = []   # 保存简单特征 (mean, std)
    all_targets = [] # 保存标签 k（注意：这里是 log+standardize 后的 k）

    # 4. 遍历整个 train_loader，提取简单特征
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            imgs = batch["img"]   # Tensor, shape: (B, C, H, W)
            ks = batch["k"]       # Tensor, shape: (B,) 或 (B, 1)

            # 展平空间维度，计算每张图的均值和方差
            B = imgs.shape[0]
            imgs_flat = imgs.view(B, -1).float()

            mean_feat = imgs_flat.mean(dim=1).cpu().numpy()  # (B,)
            std_feat = imgs_flat.std(dim=1, unbiased=False).cpu().numpy()  # (B,)

            # 这里用 2 维特征: [mean, std]
            x_batch = np.stack([mean_feat, std_feat], axis=1)  # (B, 2)

            y_batch = ks.view(-1).cpu().numpy()  # (B,)

            all_feats.append(x_batch)
            all_targets.append(y_batch)

            if i == 0:
                print("[DEBUG] 第一批特征示例:")
                print("        mean_feat:", mean_feat[:5])
                print("        std_feat :", std_feat[:5])
                print("        k(after log+std):", y_batch[:5])

    # 5. 堆叠成完整数组
    X = np.concatenate(all_feats, axis=0)   # (N, 2)
    y = np.concatenate(all_targets, axis=0) # (N,)

    print("[INFO] 总样本数:", X.shape[0])
    print("[INFO] 特征维度:", X.shape[1])

    # 6. 拟合简单线性模型，计算 RMSE & R²
    rmse_simple, r2_simple, w = fit_linear_and_r2(X, y)

    # 7. 计算“只预测均值”的常数基线表现
    y_mean_pred = np.full_like(y, y.mean())
    rmse_mean = np.sqrt(np.mean((y - y_mean_pred) ** 2))
    ss_res_mean = np.sum((y - y_mean_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_mean = 1.0 - ss_res_mean / ss_tot if ss_tot > 0 else float("nan")

    print("\n================= 简单特征线性回归结果 =================")
    print(f"  线性模型: y ≈ w0 + w1*mean + w2*std")
    print(f"    w = {w}")
    print(f"    RMSE_simple = {rmse_simple:.4f}")
    print(f"    R2_simple   = {r2_simple:.4f}")
    print("----------------------------------------------------")
    print("  常数均值基线 (总是预测 y 的均值):")
    print(f"    RMSE_mean = {rmse_mean:.4f}")
    print(f"    R2_mean   = {r2_mean:.4f}  (理论上接近 0)")
    print("====================================================\n")

    print("结论解读建议：")
    print("  - 如果 R2_simple 明显 > 0（比如 0.2/0.3 以上），说明图像的全局亮度/对比度里确实带有渗透率信息；")
    print("  - 如果 R2_simple ≈ 0，甚至是负的，说明这些非常粗糙的特征几乎和 k 无关；")
    print("  - 若你后面 Swin 训练出的 R² 也长期 ~0，那就要强烈怀疑：图像和标签有没有对齐问题。")


if __name__ == "__main__":
    main()
