# tools/predict_temporal.py
# ------------------------------------------------------------
# 时序模型推理脚本
# 改动：
# - 移除所有校准/标定逻辑与参数（--calibrate/--calib-plot/--apply-calib）
# - 不再生成 k_pred_cal / calibration_fit.png / *_cal.csv
# - 默认输出目录改为 outputs/infer_temTVP
# 仍生成：
# - 预测CSV、对比CSV
# - 图：kt_pred/kt_pred_gt/kt_seq、散点、误差散点、误差直方图
# - 若有阶段头：相变时刻误差直方图
# ------------------------------------------------------------
import os
import sys
import argparse
import time
import math
import copy
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.train_utils import file2dict
from models.build import BuildNet
from utils.dataloader_seq import SequenceDataset, collate_seq
from utils.history import History
from core.datasets.compose import Compose


def _resolve_ckpt_path(ckpt_arg):
    p = os.path.abspath(ckpt_arg)
    if os.path.isfile(p):
        return p
    matches = glob.glob(p)
    if not matches and os.path.isdir(p):
        matches = glob.glob(os.path.join(p, "*.pth"))
    if not matches:
        raise FileNotFoundError(f'No checkpoint matched: {ckpt_arg}')
    matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return matches[0]


def _find_label_norm_and_log(pipeline):
    mean, std = None, None
    log_base, offset = None, 0.0
    for step in pipeline:
        if isinstance(step, dict) and step.get('type') == 'StandardizeLabels':
            mean = float(step.get('mean')); std = float(step.get('std'))
        if isinstance(step, dict) and step.get('type') == 'LogTransform':
            log_base = step.get('log_base', math.e); offset = float(step.get('offset', 0.0))
    return mean, std, log_base, offset


def _inverse_transform(arr, mean, std, log_base, offset):
    x = arr.detach().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
    if mean is not None and std is not None:
        x = x * std + mean
    if log_base is None:
        k = x
    else:
        base = float(log_base) if isinstance(log_base, (int, float)) else math.e
        k = np.power(base, x) - float(offset)
    return k.astype(np.float32)


def _first_table_in_dir(d):
    d = os.path.abspath(d)
    if not os.path.isdir(d):
        raise FileNotFoundError('Directory not found: {}'.format(d))
    files = [f for f in os.listdir(d) if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
    if len(files) == 0:
        raise FileNotFoundError('No xlsx/csv file found in {}'.format(d))
    if len(files) == 1:
        return os.path.join(d, files[0])
    prefer = ('infer', 'predict', 'unlabeled', 'right', 'test', 'valid', 'val')
    scored = []
    for fname in files:
        name = fname.lower()
        score = 0
        for k in prefer:
            if k in name:
                score += 1
        scored.append((score, fname))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return os.path.join(d, scored[0][1])


def _load_table(p):
    p = os.path.abspath(p)
    if p.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(p)
    return pd.read_csv(p)


def _ensure_filename(df):
    if 'filename' in df.columns:
        return df
    if 'index' in df.columns:
        df = df.copy()
        df['filename'] = df['index'].apply(lambda x: f'{int(x)}.jpg' if pd.notna(x) else None)
        return df
    raise KeyError("Need 'filename' or 'index' column in the table.")


def _ensure_seq_id_t(df):
    df = df.copy()
    if 'seq_id' not in df.columns:
        df['seq_id'] = 0
    if 't' not in df.columns:
        if 'filename' in df.columns and df['filename'].notna().all():
            try:
                df = df.sort_values(by='index')
            except Exception:
                pass
        df['t'] = np.arange(len(df), dtype=int)
    return df


def _ensure_tvp(df, constT=None, constV=None, constP=None):
    df = df.copy()
    for name, val in [('T', constT), ('V', constV), ('P', constP)]:
        if name not in df.columns:
            fill = 0.0 if val is None else float(val)
            df[name] = fill
    return df


def _sanitize_pair(y_pred, y_true):
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]
    true = true[mask]
    return pred.astype(np.float32), true.astype(np.float32)


def _fallback_scatter(y_true, y_pred, path_png, title='GT vs Pred'):
    if y_true.size == 0 or y_pred.size == 0:
        return
    x = y_true
    y = y_pred
    vmin = float(min(np.min(x), np.min(y)))
    vmax = float(max(np.max(x), np.max(y)))
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, alpha=0.6, edgecolors='none')
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5)
    plt.xlabel('Ground truth k'); plt.ylabel('Predicted k'); plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def _fallback_error_plots(y_true, y_pred, path_scatter, path_hist):
    if y_true.size == 0 or y_pred.size == 0:
        return
    err = (y_pred - y_true)
    idx = np.arange(err.size)

    plt.figure(figsize=(8, 3))
    plt.scatter(idx, err, s=8, alpha=0.6, edgecolors='none')
    plt.axhline(0.0, color='r', linestyle='--', linewidth=1.0)
    plt.xlabel('Index'); plt.ylabel('Error (pred - gt)'); plt.title('Error Scatter')
    plt.tight_layout()
    plt.savefig(path_scatter, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=50, alpha=0.8, color='steelblue', edgecolor='none')
    plt.xlabel('Error'); plt.ylabel('Count'); plt.title('Error Histogram')
    plt.tight_layout()
    plt.savefig(path_hist, dpi=150)
    plt.close()


def _compose_no_collect(pipeline_cfg):
    steps = [st for st in pipeline_cfg if not (isinstance(st, dict) and st.get('type') == 'Collect')]
    return Compose(steps)


def _first_one(arr: np.ndarray) -> int:
    idx = np.where(arr.astype(int) == 1)[0]
    return int(idx[0]) if idx.size > 0 else -1


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
        save_path = os.path.join(out_dir, f'{prefix}_seq{sid}.png')
        try:
            plt.savefig(save_path, dpi=150)
        finally:
            plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description='Predict on infer, compare with right and draw figures (temporal)')
    ap.add_argument('config', help='temporal config path (e.g., models/swin_transformer/tiny_224_temporal.py)')
    ap.add_argument('--ckpt', required=True, help='checkpoint path (.pth/.pt or directory or wildcard)')
    ap.add_argument('--infer-root', default='datasetone3/infer', help='folder that contains infer xlsx/csv and images')
    ap.add_argument('--infer-csv', default=None, help='infer table path (xlsx/csv), optional')
    ap.add_argument('--right-root', default='datasetone3/right', help='folder that contains right xlsx/csv')
    ap.add_argument('--right-csv', default=None, help='right table path (xlsx/csv), optional')
    ap.add_argument('--right-k-col', default='k', help='column name of ground truth k in right table')
    ap.add_argument('--seq-len', type=int, default=1, help='temporal window length (default 1: row-aligned)')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--gpu-id', type=int, default=0)
    ap.add_argument('--return-stage-prob', action='store_true', help='also return stage probabilities')
    ap.add_argument('--out-csv', default=None, help='output csv path for predictions（默认 outputs/infer_temTVP）')
    ap.add_argument('--out-compare', default=None, help='output csv path for comparison（默认同目录）')
    ap.add_argument('--const-T', type=float, default=None)
    ap.add_argument('--const-V', type=float, default=None)
    ap.add_argument('--const-P', type=float, default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    model_cfg, train_pipeline, valid_pipeline, test_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    infer_pipe = copy.deepcopy(test_pipeline if test_pipeline else valid_pipeline)
    mean, std, log_base, offset = _find_label_norm_and_log(infer_pipe)
    print(f'Label StdConfig -> mean={mean}, std={std}, log_base={log_base}, offset={offset}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BuildNet(model_cfg).to(device)
    if device.type == 'cuda':
        model = DataParallel(model, device_ids=[args.gpu_id])

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    print('Using checkpoint:', ckpt_path)
    try:
        raw = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f'Failed to load ckpt: {e}')
    state = raw.get('state_dict', raw)
    model_sd = model.state_dict()
    new_sd, matched = {}, 0
    for k, v in state.items():
        k2 = k[7:] if k.startswith('module.') else k
        if k2 in model_sd:
            new_sd[k2] = v; matched += 1
        else:
            km = 'module.' + k2
            if km in model_sd:
                new_sd[km] = v; matched += 1
    model.load_state_dict(new_sd, strict=False)
    print(f'Checkpoint loaded. matched={matched}')

    infer_root = os.path.abspath(args.infer_root)
    if not os.path.isdir(infer_root):
        raise FileNotFoundError(f'infer-root not found: {infer_root}')
    infer_csv = args.infer_csv if args.infer_csv else _first_table_in_dir(infer_root)
    infer_csv = os.path.abspath(infer_csv)
    print('Infer table:', infer_csv)
    print('Infer images root:', infer_root)

    df_infer = _load_table(infer_csv)
    df_infer = _ensure_filename(df_infer)
    df_infer = _ensure_seq_id_t(df_infer)
    df_infer = _ensure_tvp(df_infer, args.const_T, args.const_V, args.const_P)

    seq_len = int(args.seq_len or 1)
    ds = SequenceDataset(df_infer, image_dir=infer_root, pipeline_cfg=infer_pipe, seq_len=seq_len)
    dl = DataLoader(ds, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=True, drop_last=False, collate_fn=collate_seq)

    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    base_out_dir = 'outputs/infer_temTVP'
    out_csv = args.out_csv or os.path.join(base_out_dir, f'infer_{stamp}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # sliding-window aggregation per frame
    base_df = ds.df.copy()
    idx_labels = list(base_df.index)
    N = len(base_df)
    idx_to_pos = {idx: pos for pos, idx in enumerate(idx_labels)}

    pred_sum = np.zeros(N, dtype=np.float64)
    pred_cnt = np.zeros(N, dtype=np.int32)
    prob_sum = None
    prob_cnt = None
    num_classes = None
    did_accumulate = False
    win_ptr = 0

    model.eval()
    with torch.no_grad():
        for batch in dl:
            imgs = batch['img'].to(device, dtype=torch.float32)  # [B,T,C,H,W]
            feed = {}
            for k in ('T', 'V', 'P'):
                if k in batch:
                    feed[k] = batch[k].to(device, dtype=torch.float32)
            if args.return_stage_prob:
                out = model(imgs, return_loss=False, train_statu=False, return_stage=True, **feed)
                if isinstance(out, (list, tuple)) and len(out) == 2:
                    preds_t, probs_t = out
                else:
                    preds_t, probs_t = out, None
            else:
                preds_t = model(imgs, return_loss=False, train_statu=False, **feed)
                probs_t = None

            preds = preds_t
            if torch.is_tensor(preds):
                preds = preds.detach().cpu().float().numpy()
            preds = preds.reshape(preds.shape[0], -1)  # [B,T]
            B, Tlen = preds.shape

            if probs_t is not None:
                if torch.is_tensor(probs_t):
                    probs_np = probs_t.detach().cpu().float().numpy()  # [B,T,C]
                else:
                    probs_np = np.asarray(probs_t, dtype=np.float32)
                _, _, Cp = probs_np.shape
                if num_classes is None:
                    num_classes = int(Cp)
                    prob_sum = np.zeros((N, num_classes), dtype=np.float64)
                    prob_cnt = np.zeros(N, dtype=np.int32)
            else:
                probs_np = None

            for b in range(B):
                if win_ptr + b >= len(ds.seqs):
                    break
                did_accumulate = True
                win_indices = ds.seqs[win_ptr + b]  # list of df index labels
                for t_local in range(Tlen):
                    if t_local >= len(win_indices):
                        break
                    idx_label = win_indices[t_local]
                    pos = idx_to_pos.get(idx_label, None)
                    if pos is None:
                        continue
                    pred_sum[pos] += float(preds[b, t_local])
                    pred_cnt[pos] += 1
                    if probs_np is not None:
                        prob_sum[pos, :] += probs_np[b, t_local, :].astype(np.float64)
                        prob_cnt[pos] += 1
            win_ptr += B

    def _framewise_infer():
        preds_std = np.zeros(N, dtype=np.float64)
        probs_avg = None
        pipe_fw = _compose_no_collect(infer_pipe)
        model.eval()
        with torch.no_grad():
            for i in range(N):
                row = base_df.iloc[i]
                data = dict(img_prefix=infer_root, img_info=dict(filename=str(row['filename'])))
                for kkey in ('T', 'V', 'P'):
                    if kkey in row and pd.notna(row[kkey]):
                        data[kkey] = np.array(row[kkey], dtype=np.float32)
                res = pipe_fw(copy.deepcopy(data))
                img = res['img'] if torch.is_tensor(res['img']) else torch.as_tensor(res['img'])
                x_seq = img.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)  # [1,1,C,H,W]
                T_seq = torch.as_tensor([[row.get('T', 0.0)]], dtype=torch.float32, device=device)
                V_seq = torch.as_tensor([[row.get('V', 0.0)]], dtype=torch.float32, device=device)
                P_seq = torch.as_tensor([[row.get('P', 0.0)]], dtype=torch.float32, device=device)
                if args.return_stage_prob:
                    out = model(x_seq, T=T_seq, V=V_seq, P=P_seq, return_loss=False, train_statu=False, return_stage=True)
                    if isinstance(out, (list, tuple)) and len(out) == 2:
                        pred_t, prob_t = out
                    else:
                        pred_t, prob_t = out, None
                else:
                    pred_t = model(x_seq, T=T_seq, V=V_seq, P=P_seq, return_loss=False, train_statu=False)
                    prob_t = None
                pv = float(pred_t.reshape(-1)[0].detach().cpu().item()) if torch.is_tensor(pred_t) else float(np.asarray(pred_t).reshape(-1)[0])
                preds_std[i] = pv
                if prob_t is not None:
                    prob_np = prob_t.reshape(-1).detach().cpu().float().numpy() if torch.is_tensor(prob_t) else np.asarray(prob_t, dtype=np.float32).reshape(-1)
                    if probs_avg is None:
                        num_c = int(prob_np.shape[0])
                        probs_avg = np.zeros((N, num_c), dtype=np.float64)
                    probs_avg[i, :] = prob_np.astype(np.float64)
        return preds_std, probs_avg

    if (not did_accumulate) or (int(pred_cnt.sum()) == 0):
        print('No sliding-window aggregation happened; falling back to frame-wise (T=1) inference.')
        preds_std_per_frame, prob_avg = _framewise_infer()
    else:
        with np.errstate(invalid='ignore'):
            preds_std_per_frame = np.divide(pred_sum, pred_cnt, where=(pred_cnt > 0))
        prob_avg = None
        if num_classes is not None and prob_sum is not None and prob_cnt is not None:
            with np.errstate(invalid='ignore'):
                prob_avg = np.divide(prob_sum, prob_cnt[:, None], where=(prob_cnt[:, None] > 0))

    if mean is not None and std is not None:
        preds_org_per_frame = _inverse_transform(preds_std_per_frame, mean, std, log_base, offset)
    else:
        preds_org_per_frame = preds_std_per_frame.copy()

    out_pred = base_df.copy()
    out_pred['pred_k_std'] = preds_std_per_frame.astype(np.float32)
    out_pred['pred_k'] = preds_org_per_frame.astype(np.float32)

    if prob_avg is not None:
        num_classes = prob_avg.shape[1]
        for c in range(num_classes):
            out_pred[f'prob_C{c}'] = prob_avg[:, c].astype(np.float32)
        if num_classes >= 2:
            out_pred['prob_CRP'] = prob_avg[:, 0].astype(np.float32)
            out_pred['prob_FRP'] = prob_avg[:, 1].astype(np.float32)
        out_pred['pred_stage'] = np.argmax(prob_avg, axis=1).astype(np.int64)

    # 写预测CSV
    if out_csv.lower().endswith(('.xlsx', '.xls')):
        out_pred.to_excel(out_csv, index=False)
    else:
        out_pred.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print('Wrote predictions to:', out_csv)

    # 图：预测曲线
    try:
        _plot_k_t_curves(out_pred, right_df=None, out_dir=os.path.dirname(out_csv), prefix='kt_pred')
    except Exception as e:
        print('Warn: plot pred k(t) failed:', e)

    # 读取 right，并绘制预测+GT曲线
    right_root = os.path.abspath(args.right_root)
    if not os.path.isdir(right_root):
        print('Right root not found, skip compare:', right_root)
        return
    right_csv = args.right_csv if args.right_csv else _first_table_in_dir(right_root)
    right_csv = os.path.abspath(right_csv)
    if not os.path.isfile(right_csv):
        print('Right table not found, skip compare:', right_csv)
        return
    print('Right table:', right_csv)

    df_right = _load_table(right_csv)
    df_right = _ensure_filename(df_right)
    df_right = _ensure_seq_id_t(df_right)

    try:
        r_for_plot = df_right.copy()
        if 'k' not in r_for_plot.columns and args.right_k_col in r_for_plot.columns:
            r_for_plot = r_for_plot.rename(columns={args.right_k_col: 'k'})
        _plot_k_t_curves(out_pred, right_df=r_for_plot, out_dir=os.path.dirname(out_csv), prefix='kt_pred_gt')
    except Exception as e:
        print('Warn: plot pred vs gt k(t) failed:', e)

    # 对齐并评估
    candidate_keys = [
        (['seq_id', 't'], 'seq_t'),
        (['filename'], 'filename'),
        (['index'], 'index')
    ]
    best = None
    for keys, tag in candidate_keys:
        if all(k in out_pred.columns for k in keys) and all(k in df_right.columns for k in keys):
            m = pd.merge(out_pred, df_right, on=keys, how='inner', suffixes=('', '_right'))
            valid = np.isfinite(pd.to_numeric(m[args.right_k_col], errors='coerce').to_numpy()).sum()
            score = valid
            if best is None or score > best['score']:
                best = dict(keys=keys, tag=tag, merged=m, score=score)
    if best is None or len(best['merged']) == 0:
        print('No overlap between predictions and right under all key modes.')
        return
    merged = best['merged']
    print('Join mode:', best['tag'], 'overlap rows:', len(merged))

    merged[args.right_k_col] = pd.to_numeric(merged[args.right_k_col], errors='coerce')

    y_pred_raw = merged['pred_k'].to_numpy()
    y_true_raw = merged[args.right_k_col].to_numpy()
    y_pred, y_true = _sanitize_pair(y_pred_raw, y_true_raw)

    print(f'Overlapped rows: {len(merged)}, valid numeric pairs: {y_true.size}')
    if y_true.size == 0:
        print('No valid numeric pairs after sanitizing, skip drawing.')
        debug_path = os.path.join(os.path.dirname(out_csv), 'infer_vs_right_debug_head.csv')
        merged.head(50).to_csv(debug_path, index=False, encoding='utf-8-sig')
        print('Dumped merged head to:', debug_path)
        return

    print('pred_k stats: min/mean/max = {:.4f} / {:.4f} / {:.4f}'.format(
        float(np.nanmin(y_pred)), float(np.nanmean(y_pred)), float(np.nanmax(y_pred))))
    print('true_k stats: min/mean/max = {:.4f} / {:.4f} / {:.4f}'.format(
        float(np.nanmin(y_true)), float(np.nanmean(y_true)), float(np.nanmax(y_true))))

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    r2 = float('nan')
    var = float(np.var(y_true))
    if y_true.size > 1 and var > 1e-12:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    print(f'Compare RMSE={rmse:.6f}, R2={r2:.6f}, samples={y_true.size}')

    out_dir = os.path.dirname(out_csv)
    meta = {
        'predicted_values': torch.as_tensor(y_pred).view(-1, 1),
        'true_values': torch.as_tensor(y_true).view(-1, 1),
        'save_dir': out_dir
    }
    hist = History(out_dir)
    scatter_path = os.path.join(out_dir, 'infer_vs_right_scatter.png')
    err_scatter = os.path.join(out_dir, 'infer_vs_right_error_scatter.png')
    err_hist = os.path.join(out_dir, 'infer_vs_right_error_hist.png')
    try:
        hist.draw_test_results(meta, scatter_path)
    except Exception as e:
        _fallback_scatter(y_true, y_pred, scatter_path, title='GT vs Pred (infer vs right)')
        print('History.draw_test_results failed, fallback scatter used:', e)
    try:
        hist.draw_error_plots(meta, err_scatter, err_hist)
    except Exception as e:
        _fallback_error_plots(y_true, y_pred, err_scatter, err_hist)
        print('History.draw_error_plots failed, fallback error plots used:', e)

    out_compare = args.out_compare or os.path.join(out_dir, f'infer_vs_right_{stamp}.csv')
    merged_out = merged.copy()
    merged_out['pred_k_used'] = y_pred
    merged_out['rmse_all'] = rmse
    merged_out['r2_all'] = r2
    if out_compare.lower().endswith(('.xlsx', '.xls')):
        merged_out.to_excel(out_compare, index=False)
    else:
        merged_out.to_csv(out_compare, index=False, encoding='utf-8-sig')
    print('Wrote comparison to:', out_compare)

    # 最终每序列曲线（kt_seq*.png）
    try:
        for sid, gdf in out_pred.groupby('seq_id'):
            gdf = gdf.sort_values('t')
            if gdf.empty:
                continue
            t_idx = gdf['t'].to_numpy(dtype=np.int64)
            k_pred_used = gdf['pred_k'].to_numpy(dtype=np.float64)

            plt.figure(figsize=(7.5, 4.2))
            plt.plot(t_idx, k_pred_used, '-o', ms=3, lw=1.6, label='Temporal (agg)')

            rs = df_right[df_right['seq_id'] == sid].copy()
            if len(rs) > 0 and args.right_k_col in rs.columns:
                rs = rs.sort_values('t')
                ygt = pd.to_numeric(rs[args.right_k_col], errors='coerce').to_numpy(dtype=np.float64)
                t_gt = pd.to_numeric(rs['t'], errors='coerce').to_numpy(dtype=np.int64)
                mask = np.isfinite(ygt) & np.isfinite(t_gt)
                if mask.any():
                    plt.plot(t_gt[mask], ygt[mask], '-o', ms=3, lw=1.6, label='GT')

            plt.xlabel('t'); plt.ylabel('k'); plt.title(f'k(t) – seq_id={sid}')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'kt_seq{sid}.png'), dpi=150)
            plt.close()
    except Exception as e:
        print('Warn: making Fig1 (agg k(t) curves) failed:', e)

    # 相变时刻误差直方图（若有阶段）
    try:
        if 'pred_stage' in out_pred.columns:
            err_list = []
            for sid, gdf in out_pred.groupby('seq_id'):
                gdf = gdf.sort_values('t')
                pred_stage_seq = gdf['pred_stage'].to_numpy(dtype=int)
                t_pred = np.where(pred_stage_seq.astype(int) == 1)[0]
                t_pred = int(t_pred[0]) if t_pred.size > 0 else -1

                rs = df_right[df_right['seq_id'] == sid].sort_values('t')
                t_gt = -1
                if 'stage' in rs.columns:
                    try:
                        gt_stage = pd.to_numeric(rs['stage'], errors='coerce').to_numpy()
                        gt_stage = np.nan_to_num(gt_stage, nan=0.0)
                        idx = np.where(gt_stage.astype(int) == 1)[0]
                        t_gt = int(idx[0]) if idx.size > 0 else -1
                    except Exception:
                        t_gt = -1

                if t_pred >= 0 and t_gt >= 0:
                    err_list.append(t_pred - t_gt)

            if len(err_list) > 0:
                errs = np.asarray(err_list, dtype=np.float64)
                p50 = float(np.percentile(np.abs(errs), 50))
                p90 = float(np.percentile(np.abs(errs), 90))
                plt.figure(figsize=(6.5, 4))
                plt.hist(errs, bins=31, alpha=0.85, color='steelblue', edgecolor='none')
                plt.axvline(0.0, color='r', linestyle='--', linewidth=1.2)
                plt.title(f'Transition time error (pred-gt); |err| P50={p50:.2f}, P90={p90:.2f}')
                plt.xlabel('t_pred - t_gt'); plt.ylabel('count')
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(out_csv), 'transition_time_error_hist.png'), dpi=150)
                plt.close()
                print(f'Transition P50={p50:.3f}, P90={p90:.3f}, samples={len(errs)}')
            else:
                print('Warn: no transition pairs found for transition histogram.')
    except Exception as e:
        print('Warn: making transition histogram failed:', e)


if __name__ == '__main__':
    main()
