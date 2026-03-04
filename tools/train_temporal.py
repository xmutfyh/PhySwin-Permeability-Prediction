# tools/train_temporal.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import time
import argparse
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch.optim as optim

from utils.train_utils import (
    train_temporal, validation_temporal, print_info, file2dict,
    init_random_seed, set_random_seed, resume_model, test_temporal
)
from utils.inference import init_model
from models.build import BuildNet
from core.optimizers.lr_update import (
    StepLrUpdater, LrUpdater, PolyLrUpdater,
    CosineAnnealingLrUpdater, CosineAnnealingCooldownLrUpdater,
    ReduceLROnPlateauLrUpdater
)
from utils.dataloader_seq import SequenceDataset, collate_seq
from utils.history import History

def parse_args():
    ap = argparse.ArgumentParser(description='Train temporal model (K(t) + stage classification)')
    ap.add_argument('config', help='config file path (temporal)')
    ap.add_argument('--data-root', default=None, help='dataset root with train/valid/test subfolders')
    ap.add_argument('--train-csv', default=None)
    ap.add_argument('--valid-csv', default=None)
    ap.add_argument('--test-csv',  default=None)
    ap.add_argument('--train-root', default=None)
    ap.add_argument('--valid-root', default=None)
    ap.add_argument('--test-root',  default=None)
    ap.add_argument('--mix-right', action='store_true')
    ap.add_argument('--right-root', default='datasetone3/right')
    ap.add_argument('--right-csv', default=None)
    ap.add_argument('--right-train-frac', type=float, default=0.6)
    ap.add_argument('--right-val-frac', type=float, default=0.2)
    ap.add_argument('--right-seed', type=int, default=2025)
    ap.add_argument('--resume-from', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--gpu-id', type=int, default=0)
    ap.add_argument('--acc-steps', type=int, default=1, help='gradient accumulation steps')
    ap.add_argument('--test-only', action='store_true', help='run only test phase and exit')
    return ap.parse_args()

def load_table(p):
    p = os.path.abspath(p)
    if p.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(p)
    return pd.read_csv(p)

def _first_table_in_dir(d, split_name):
    d = os.path.abspath(d)
    if not os.path.isdir(d):
        raise FileNotFoundError('Directory not found: {}'.format(d))
    files = [f for f in os.listdir(d) if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
    if len(files) == 0:
        raise FileNotFoundError('No xlsx/csv file found in {}'.format(d))
    if len(files) == 1:
        return os.path.join(d, files[0])
    key_map = {'train': ('train',), 'valid': ('valid', 'val', 'validation'), 'test':  ('test', 'eval'), 'right': ('right', 'normal', 'gt', 'label')}
    keys = key_map.get(split_name, (split_name,))
    scored = []
    for fname in files:
        name = fname.lower()
        score = sum(1 for k in keys if k in name)
        scored.append((score, fname))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return os.path.join(d, scored[0][1])

def _find_split_dir(root, candidates):
    root = os.path.abspath(root)
    for name in candidates:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    try:
        entries = os.listdir(root)
    except Exception:
        return None
    lower_map = {e.lower(): e for e in entries}
    for name in candidates:
        low = name.lower()
        if low in lower_map:
            p = os.path.join(root, lower_map[low])
            if os.path.isdir(p):
                return p
    return None

def _resolve_from_data_root(data_root):
    if data_root is None:
        return None
    root = os.path.abspath(data_root)
    train_dir = _find_split_dir(root, ('train',))
    valid_dir = _find_split_dir(root, ('valid', 'val', 'validation'))
    test_dir  = _find_split_dir(root, ('test', 'eval'))
    if train_dir is None or valid_dir is None or test_dir is None:
        raise FileNotFoundError('Cannot find train/valid/test subfolders under {}'.format(root))
    train_csv = _first_table_in_dir(train_dir, 'train')
    valid_csv = _first_table_in_dir(valid_dir, 'valid')
    test_csv  = _first_table_in_dir(test_dir,  'test')
    return dict(train_csv=train_csv, valid_csv=valid_csv, test_csv=test_csv,
                train_root=train_dir, valid_root=valid_dir, test_root=test_dir)

def _ensure_seq_id_t(df):
    df = df.copy()
    if 'seq_id' not in df.columns:
        df['seq_id'] = 0
    if 't' not in df.columns:
        df['t'] = np.arange(len(df), dtype=int)
    return df

def align_sparse_k(base_df, right_df, prefer_keys=(('seq_id','t'), ('filename',), ('index',)), k_col='k'):
    out = base_df.copy()
    r = right_df.copy()
    if k_col not in r.columns:
        raise KeyError(f'right表缺少列 {k_col}')
    for keys in prefer_keys:
        if all(k in out.columns for k in keys) and all(k in r.columns for k in keys):
            merged = out.merge(r[list(keys) + [k_col]], on=list(keys), how='left', suffixes=('', '_r'))
            if f'{k_col}_r' in merged.columns:
                merged[k_col] = merged[f'{k_col}_r'].where(pd.notna(merged[f'{k_col}_r']), merged.get(k_col, np.nan))
                merged = merged.drop(columns=[f'{k_col}_r'])
            return merged
    return out

def main():
    args = parse_args()
    model_cfg, train_pipeline, valid_pipeline, test_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    print_info(model_cfg)

    if args.data_root:
        paths = _resolve_from_data_root(args.data_root)
    else:
        if not (args.train_csv and args.valid_csv and args.test_csv):
            raise ValueError('Either use --data-root, or provide --train-csv/--valid-csv/--test-csv together.')
        paths = dict(
            train_csv=os.path.abspath(args.train_csv),
            valid_csv=os.path.abspath(args.valid_csv),
            test_csv=os.path.abspath(args.test_csv),
            train_root=os.path.abspath(args.train_root) if args.train_root else os.path.dirname(os.path.abspath(args.train_csv)),
            valid_root=os.path.abspath(args.valid_root) if args.valid_root else os.path.dirname(os.path.abspath(args.valid_csv)),
            test_root=os.path.abspath(args.test_root) if args.test_root else os.path.dirname(os.path.abspath(args.test_csv)),
        )

    print('Resolved dataset paths:')
    print('  train_csv :', paths['train_csv'])
    print('  valid_csv :', paths['valid_csv'])
    print('  test_csv  :', paths['test_csv'])
    print('  train_root:', paths['train_root'])
    print('  valid_root:', paths['valid_root'])
    print('  test_root :', paths['test_root'])

    meta = dict()
    dirname = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    save_dir = os.path.join('logs', model_cfg.get('backbone').get('type'), dirname)
    meta['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=True)
    meta['seed'] = seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', 'GPU ' + torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')

    print('Build model.')
    model = BuildNet(model_cfg)
    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()
    if device.type == 'cuda':
        model = DataParallel(model, device_ids=[args.gpu_id])

    seq_len = int(data_cfg.get('seq_len', 8))
    train_df = load_table(paths['train_csv'])
    valid_df = load_table(paths['valid_csv'])
    test_df  = load_table(paths['test_csv'])

    if args.mix_right:
        right_root = os.path.abspath(args.right_root) if args.right_root else None
        if right_root is None or not os.path.isdir(right_root):
            print('Warn: right-root not found, skip sparse mixing:', right_root)
        else:
            right_csv = args.right_csv if args.right_csv else _first_table_in_dir(right_root, 'right')
            right_csv = os.path.abspath(right_csv)
            print('Right root:', right_root); print('Right csv :', right_csv)
            df_right = load_table(right_csv)
            df_right = _ensure_seq_id_t(df_right)
            train_df = align_sparse_k(_ensure_seq_id_t(train_df), df_right, k_col='k')
            valid_df = align_sparse_k(_ensure_seq_id_t(valid_df), df_right, k_col='k')

    train_main_ds = SequenceDataset(train_df, image_dir=paths['train_root'], pipeline_cfg=train_pipeline, seq_len=seq_len)
    valid_main_ds = SequenceDataset(valid_df, image_dir=paths['valid_root'], pipeline_cfg=valid_pipeline, seq_len=seq_len)
    test_ds       = SequenceDataset(test_df,  image_dir=paths['test_root'],  pipeline_cfg=test_pipeline,  seq_len=seq_len)

    train_loader = DataLoader(train_main_ds, shuffle=True,
                              batch_size=data_cfg.get('batch_size'),
                              num_workers=data_cfg.get('num_workers'),
                              pin_memory=True, drop_last=True,
                              collate_fn=collate_seq)
    val_loader = DataLoader(valid_main_ds, shuffle=False,
                            batch_size=data_cfg.get('batch_size'),
                            num_workers=data_cfg.get('num_workers'),
                            pin_memory=True, drop_last=True,
                            collate_fn=collate_seq)
    test_loader = DataLoader(test_ds, shuffle=False,
                             batch_size=data_cfg.get('batch_size'),
                             num_workers=data_cfg.get('num_workers'),
                             pin_memory=True, drop_last=False,
                             collate_fn=collate_seq)

    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)

    robust_cfg = (data_cfg.get('train') or {}).get('robust', {}) or {}
    runner = dict(
        optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        iter=0, epoch=0,
        max_epochs=data_cfg.get('train').get('epoches'),
        max_iters=data_cfg.get('train').get('epoches') * max(1, len(train_loader)),
        best_train_loss=float('INF'), best_val_rmse=float('INF'),
        best_train_weight='', best_val_weight='', last_weight='',
        robust_cfg=robust_cfg, groupdro_state=dict(q={}),
        accumulate_steps=int(args.acc_steps)
    )
    runner['config_path'] = args.config
    runner['test_pipeline'] = test_pipeline

    meta['train_info'] = dict(train_loss=[], val_loss=[], train_metric=[], val_metric=[])
    meta['test_info']  = dict(test_loss=[], test_metric=[])

    if args.resume_from:
        model, runner, meta = resume_model(model, runner, args.resume_from, meta)
    else:
        shutil.copyfile(args.config, os.path.join(save_dir, os.path.split(args.config)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')

    train_history = History(meta['save_dir'])

    if args.test_only:
        test_temporal(model, runner, data_cfg.get('test'), device, meta)
        return

    lr_update_func.before_run(runner)
    for epoch in range(runner.get('epoch'), runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)
        train_temporal(model, runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'),
                       data_cfg.get('test'), meta)
        validation_temporal(model, runner, data_cfg.get('test'), device, epoch,
                            data_cfg.get('train').get('epoches'), meta)
        torch.cuda.empty_cache()
        train_history.after_epoch(meta)

    test_temporal(model, runner, data_cfg.get('test'), device, meta)

if __name__ == '__main__':
    main()
