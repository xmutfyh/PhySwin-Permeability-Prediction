# tools/predict_static_seq.py
# ------------------------------------------------------------
# 单帧静态模型推理（支持 physwin: img+TVP；支持 image-only: img only）
# 说明：
# - image-only：不要加 --use-tvp
# - physwin：加 --use-tvp
# ------------------------------------------------------------
import os
import sys
import glob
import argparse
import runpy
import math
import time
import numpy as np
import random
import pandas as pd
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seed for reproducible inference (default: 42)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from core.datasets.compose import Compose
from models.build import BuildNet
from utils.checkpoint import load_checkpoint


def norm_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_col(df, candidates, must=True, name=""):
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    norm = {c.lower().replace(" ", ""): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        if key in norm:
            return norm[key]
    if must:
        raise ValueError(f"找不到{name}列。候选={candidates}\n实际列={cols}")
    return None

def _sanitize_pair(y_pred, y_true):
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(t)
    return p[m].astype(np.float32), t[m].astype(np.float32)

def _fallback_scatter(y_true, y_pred, path_png, title='GT vs Pred'):
    if y_true.size == 0 or y_pred.size == 0:
        return
    vmin = float(min(np.min(y_true), np.min(y_pred)))
    vmax = float(max(np.max(y_true), np.max(y_pred)))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6, edgecolors='none')
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5)
    plt.xlabel('Ground truth k'); plt.ylabel('Predicted k'); plt.title(title)
    plt.tight_layout(); plt.savefig(path_png, dpi=150); plt.close()

def _fallback_error_plots(y_true, y_pred, path_scatter, path_hist):
    if y_true.size == 0 or y_pred.size == 0:
        return
    err = (y_pred - y_true); idx = np.arange(err.size)
    plt.figure(figsize=(8, 3))
    plt.scatter(idx, err, s=8, alpha=0.6, edgecolors='none')
    plt.axhline(0.0, color='r', linestyle='--', linewidth=1.0)
    plt.xlabel('Index'); plt.ylabel('Error (pred - gt)'); plt.title('Error Scatter')
    plt.tight_layout(); plt.savefig(path_scatter, dpi=150); plt.close()
    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=50, alpha=0.8, color='steelblue', edgecolor='none')
    plt.xlabel('Error'); plt.ylabel('Count'); plt.title('Error Histogram')
    plt.tight_layout(); plt.savefig(path_hist, dpi=150); plt.close()

def _plot_k_t_curves(pred_df, right_df=None, out_dir='.', prefix='kt'):
    os.makedirs(out_dir, exist_ok=True)
    p = pred_df.copy()
    if not {'seq_id', 't', 'pred_k'} <= set(p.columns):
        return
    p = p[['seq_id', 't', 'pred_k']].dropna()
    p['t'] = pd.to_numeric(p['t'], errors='coerce')
    p['pred_k'] = pd.to_numeric(p['pred_k'], errors='coerce')
    p = p[np.isfinite(p['t']) & np.isfinite(p['pred_k'])]

    r = None
    if right_df is not None and {'seq_id', 't', 'k'} <= set(right_df.columns):
        r = right_df[['seq_id', 't', 'k']].copy()
        r['t'] = pd.to_numeric(r['t'], errors='coerce')
        r['k'] = pd.to_numeric(r['k'], errors='coerce')
        r = r[np.isfinite(r['t']) & np.isfinite(r['k'])]

    for sid, ps in p.groupby('seq_id'):
        ps = ps.sort_values('t')
        if ps.empty:
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(ps['t'].values, ps['pred_k'].values, '-o', ms=3, lw=1.5, label='Pred k(t)')
        if r is not None:
            rs = r[r['seq_id'] == sid].sort_values('t')
            if not rs.empty:
                plt.plot(rs['t'].values, rs['k'].values, '-o', ms=3, lw=1.5, label='GT k(t)')
        plt.xlabel('t'); plt.ylabel('k'); plt.title(f'k(t) – seq_id={sid}')
        plt.legend(loc='best'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{prefix}_seq{sid}.png'), dpi=150)
        plt.close()

def build_meta_from_dirs(infer_dir, right_dir, use_tvp=True):
    infer_excels = glob.glob(os.path.join(infer_dir, "*.xlsx")) + glob.glob(os.path.join(infer_dir, "*.xls"))
    if not infer_excels:
        raise ValueError(f"{infer_dir} 中没找到 infer excel")
    infer_df = norm_cols(pd.read_excel(infer_excels[0]))

    idx_col = pick_col(infer_df, ["index", "idx"], True, "index")
    seq_col = pick_col(infer_df, ["seq_id", "seqid", "seq"], True, "seq_id")
    t_col   = pick_col(infer_df, ["t", "time", "frame"], True, "t")

    if use_tvp:
        T_col = pick_col(infer_df, ["T"], True, "T")
        V_col = pick_col(infer_df, ["V"], True, "V")
        P_col = pick_col(infer_df, ["P"], True, "P")
        infer_df = infer_df[[idx_col, seq_col, t_col, T_col, V_col, P_col]].copy()
        infer_df.columns = ["index", "seq_id", "t", "T", "V", "P"]
    else:
        infer_df = infer_df[[idx_col, seq_col, t_col]].copy()
        infer_df.columns = ["index", "seq_id", "t"]

    infer_df["index"] = infer_df["index"].astype(int)
    infer_df["seq_id"] = infer_df["seq_id"].astype(int)
    infer_df["t"] = infer_df["t"].astype(int)

    right_excels = glob.glob(os.path.join(right_dir, "*.xlsx")) + glob.glob(os.path.join(right_dir, "*.xls"))
    if not right_excels:
        raise ValueError(f"{right_dir} 中没找到 right excel")
    right_df = norm_cols(pd.read_excel(right_excels[0]))

    r_idx_col = pick_col(right_df, ["index", "idx"], True, "index")
    r_k_col   = pick_col(right_df, ["k", "k_true"], True, "k")
    right_df = right_df[[r_idx_col, r_k_col]].copy()
    right_df.columns = ["index", "k"]
    right_df["index"] = right_df["index"].astype(int)

    img_paths = sorted(
        glob.glob(os.path.join(infer_dir, "**", "*.jpg"), recursive=True) +
        glob.glob(os.path.join(infer_dir, "**", "*.png"), recursive=True)
    )
    if not img_paths:
        raise ValueError(f"{infer_dir} 中没找到图片")

    imgs = []
    for p in img_paths:
        fname = os.path.splitext(os.path.basename(p))[0]
        try:
            idx = int(fname)
        except Exception:
            raise ValueError(f"图片名不是数字 index：{p}")
        imgs.append({"index": idx, "img_path": p.replace("\\", "/")})
    img_df = pd.DataFrame(imgs)

    meta = img_df.merge(infer_df, on="index", how="left").merge(right_df, on="index", how="left")
    meta["filename"] = meta["img_path"].apply(lambda p: os.path.basename(p))
    return meta.sort_values(["seq_id", "t"])

def build_model_from_cfg(cfg_path, device):
    cfg_vars = runpy.run_path(cfg_path)
    model_cfg = cfg_vars["model_cfg"]
    test_pipeline = cfg_vars["test_pipeline"]
    model = BuildNet(model_cfg).to(device)
    return model, test_pipeline

def find_label_norm_and_log(pipeline):
    mean, std = None, None
    log_base, offset = None, 0.0
    for step in pipeline:
        if isinstance(step, dict) and step.get("type") == "StandardizeLabels":
            mean = float(step["mean"])
            std  = float(step["std"])
        if isinstance(step, dict) and step.get("type") == "LogTransform":
            log_base = step.get("log_base", math.e)
            offset   = float(step.get("offset", 0.0))
    return mean, std, log_base, offset

def inverse_transform(pred_std, mean, std, log_base, offset):
    x = float(pred_std.detach().cpu()) if torch.is_tensor(pred_std) else float(pred_std)
    if mean is not None and std is not None:
        x = x * std + mean
    if log_base is not None:
        x = (float(log_base) ** x) - float(offset)
    return float(x)

def infer_one(model, img, T=None, V=None, P=None, use_tvp=False):
    if use_tvp:
        return model(img, targets=None, T=T, V=V, P=P, return_loss=False, train_statu=False)
    else:
        return model(img, targets=None, return_loss=False, train_statu=False)


def apply_tvp_ablation(T, V, P, mode: str = "TVP"):
    """Zero-out selected TVP entries while keeping argument order stable (T,V,P)."""
    mode = (mode or "TVP").upper()
    if T is None and V is None and P is None:
        return T, V, P

    def _zeros_like(x, ref):
        if x is not None:
            return torch.zeros_like(x)
        if ref is not None:
            return torch.zeros_like(ref)
        return None

    if mode == "TVP":
        return T, V, P
    if mode == "TP":
        ref = T if T is not None else P
        V = _zeros_like(V, ref)
        return T, V, P
    if mode == "TV":
        ref = T if T is not None else V
        P = _zeros_like(P, ref)
        return T, V, P
    if mode == "T":
        ref = T
        V = _zeros_like(V, ref)
        P = _zeros_like(P, ref)
        return T, V, P
    raise ValueError(f"Unknown tvp ablation mode: {mode}")


def parse_args():
    ap = argparse.ArgumentParser(description="Static single-frame inference")
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--infer-dir", required=True)
    ap.add_argument("--right-dir", required=True)
    ap.add_argument("--use-tvp", action="store_true", help="physwin: img+TVP；不加则 image-only")
    ap.add_argument("--tvp-mode", default="TVP", choices=["TVP","TP","TV","T"],
                    help="TVP ablation mode (only effective when --use-tvp): TVP/TP/TV/T")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-compare", default=None)
    ap.add_argument("--cond-only", action="store_true", help="Remove image information (set img to zero), use conditions only")
    return ap.parse_args()

def main():
    # Fix random seed for reproducibility (ablation comparisons)
    set_seed(42, deterministic=True)
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Info] Using device:", device)

    model, test_pipeline_cfg = build_model_from_cfg(args.config, device)
    mean, std, log_base, offset = find_label_norm_and_log(test_pipeline_cfg)
    print(f"[Info] Label inverse: mean={mean}, std={std}, log_base={log_base}, offset={offset}")

    print("[Info] Loading checkpoint:", args.ckpt)
    load_checkpoint(model, args.ckpt, map_location=device)
    model.eval()

    test_pipeline = Compose(test_pipeline_cfg)
    df = build_meta_from_dirs(args.infer_dir, args.right_dir, use_tvp=args.use_tvp)
    has_k = "k" in df.columns

    records = []
    with torch.no_grad():
        for _, row in df.iterrows():
            sample = dict(img_prefix=None, img_info=dict(filename=row["img_path"]))
            if args.use_tvp:
                sample["T"] = float(row["T"])
                sample["V"] = float(row["V"])
                sample["P"] = float(row["P"])
            if has_k and not pd.isna(row["k"]):
                sample["k"] = float(row["k"])

            sample = test_pipeline(sample)
            img = sample["img"].unsqueeze(0).to(device)
            if args.cond_only:
                img = torch.zeros_like(img)

            if args.use_tvp:
                Tt = sample["T"].unsqueeze(0).to(device)
                Vt = sample["V"].unsqueeze(0).to(device)
                Pt = sample["P"].unsqueeze(0).to(device)
                Tt, Vt, Pt = apply_tvp_ablation(Tt, Vt, Pt, args.tvp_mode)
                pred_std = infer_one(model, img, T=Tt, V=Vt, P=Pt, use_tvp=True)
            else:
                pred_std = infer_one(model, img, use_tvp=False)

            pred_std = pred_std.view(-1)[0]
            k_pred = inverse_transform(pred_std, mean, std, log_base, offset)

            rec = {
                "filename": row["filename"],
                "index": int(row["index"]),
                "seq_id": int(row["seq_id"]),
                "t": int(row["t"]),
                "pred_k_std": float(pred_std.detach().cpu()),
                "pred_k": float(k_pred)
            }
            if args.use_tvp:
                rec["T"] = float(row["T"])
                rec["V"] = float(row["V"])
                rec["P"] = float(row["P"])
            if has_k:
                rec["k"] = row.get("k", np.nan)
            records.append(rec)

    out_pred = pd.DataFrame(records).sort_values(["seq_id", "t"])
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # Default output directory:
    # - physwin (with TVP): outputs/infer_TVP/<tvp-mode>/[condOnly]/
    # - image-only (no TVP): outputs/infer_noTVP/[condOnly]/
    if args.use_tvp:
        mode_dir = (args.tvp_mode or "TVP").upper()
        base_out_dir = os.path.join("outputs", "infer_TVP", mode_dir)
    else:
        base_out_dir = os.path.join("outputs", "infer_noTVP")

    if args.cond_only:
        base_out_dir = os.path.join(base_out_dir, "condOnly")

    default_out_csv = os.path.join(base_out_dir, f"infer_{stamp}.csv")
    out_csv = args.out_csv or default_out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_pred.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[Done] Wrote predictions to:", out_csv)

    out_dir = os.path.dirname(out_csv)

    # curves
    try:
        _plot_k_t_curves(out_pred, right_df=None, out_dir=out_dir, prefix='kt_pred')
    except Exception as e:
        print('[Warn] plot pred k(t) failed:', e)

    r_for_plot = None
    if has_k:
        try:
            r_for_plot = out_pred[['seq_id', 't', 'k']].copy()
            _plot_k_t_curves(out_pred, right_df=r_for_plot, out_dir=out_dir, prefix='kt_pred_gt')
        except Exception as e:
            print('[Warn] plot pred vs gt k(t) failed:', e)

    if has_k:
        y_pred, y_true = _sanitize_pair(out_pred['pred_k'].to_numpy(), out_pred['k'].to_numpy())
        print(f'Overlapped rows: {len(out_pred)}, valid numeric pairs: {y_true.size}')
        if y_true.size > 0:
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            r2 = float('nan')
            var = float(np.var(y_true))
            if y_true.size > 1 and var > 1e-12:
                r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            print(f'Compare RMSE={rmse:.6f}, R2={r2:.6f}, samples={y_true.size}')

            scatter_path = os.path.join(out_dir, 'infer_vs_right_scatter.png')
            err_scatter = os.path.join(out_dir, 'infer_vs_right_error_scatter.png')
            err_hist = os.path.join(out_dir, 'infer_vs_right_error_hist.png')
            _fallback_scatter(y_true, y_pred, scatter_path, title='GT vs Pred (static infer vs right)')
            _fallback_error_plots(y_true, y_pred, err_scatter, err_hist)

            default_out_compare = os.path.join(out_dir, f'infer_vs_right_{stamp}.csv')
            out_compare = args.out_compare or default_out_compare
            merged_out = out_pred.copy()
            merged_out = merged_out[np.isfinite(pd.to_numeric(merged_out['k'], errors='coerce'))]
            merged_out['rmse_all'] = rmse
            merged_out['r2_all'] = r2
            merged_out.to_csv(out_compare, index=False, encoding='utf-8-sig')
            print('Wrote comparison to:', out_compare)
        else:
            print('No valid numeric pairs after sanitizing, skip compare plots.')
    else:
        print('Right table has no k column, skip compare and GT-related figures.')

if __name__ == "__main__":
    main()
