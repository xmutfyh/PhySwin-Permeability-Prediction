import os
import random
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.train import load_labels
import argparse
import copy
from tqdm import tqdm
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from models.build import BuildNet
from utils.dataloader import MyDataset, collate
from utils.train_utils import file2dict
from core.evaluations import evaluate
import math

def parse_args():
    ap = argparse.ArgumentParser(description='Evaluate a model (robust ckpt load + dual-scale metrics)')
    ap.add_argument('config', help='config file path')
    ap.add_argument('--device', default=None)
    ap.add_argument('--gpu-id', type=int, default=0)
    return ap.parse_args()

def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def _collect_keys_from_pipeline(pipeline):
    for step in pipeline:
        if isinstance(step, dict) and step.get('type') == 'Collect':
            return list(step.get('keys', []))
    return None

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
        base = float(log_base) if isinstance(log_base, (int,float)) else math.e
        k = np.power(base, x) - float(offset)
    return torch.from_numpy(k.astype(np.float32))

def robust_load_ckpt(model, ckpt_path, device):
    print(f'Loading {os.path.basename(ckpt_path)}')
    raw = torch.load(ckpt_path, map_location=device)
    state = raw.get('state_dict', raw)

    model_sd = model.state_dict()
    new_sd = {}
    matched, missing, unexpected = 0, [], []

    # 尝试去掉或增加 module. 前缀进行匹配
    for k, v in state.items():
        k_stripped = k[7:] if k.startswith('module.') else k
        if k_stripped in model_sd:
            new_sd[k_stripped] = v
            matched += 1
        else:
            # 尝试加上 module.
            k_mod = 'module.' + k_stripped
            if k_mod in model_sd:
                new_sd[k_mod] = v
                matched += 1
            else:
                unexpected.append(k)

    # 统计缺失键
    for k in model_sd.keys():
        if k not in new_sd:
            missing.append(k)

    model.load_state_dict(new_sd, strict=False)
    print(f'CKPT load report -> matched: {matched}, missing: {len(missing)}, unexpected: {len(unexpected)}')
    if len(missing) > 0:
        # 打印一些关键层看看是否缺 head（只打印前 10 个）
        sample = missing[:10]
        print('Sample missing keys:', sample)
    if len(unexpected) > 0:
        sample = unexpected[:10]
        print('Sample unexpected keys:', sample)
    return model

def main():
    args = parse_args()
    model_cfg, train_pipeline, val_pipeline, test_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)

    set_random_seed(33)
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建未包 DataParallel 的模型，先加载权重，后视情况再并行
    model = BuildNet(model_cfg).to(device)
    ckpt = data_cfg.get('test', {}).get('ckpt', '')
    if not ckpt:
        raise ValueError('Please set data_cfg.test.ckpt in your config for evaluation.')

    model = robust_load_ckpt(model, ckpt, device)
    model.eval()

    # 使用 test_pipeline
    vp = copy.deepcopy(test_pipeline)
    keys_in_collect = _collect_keys_from_pipeline(vp)
    if keys_in_collect is None:
        keys_in_collect = ['img', 'k', 'filename']
        for k in ('T', 'V', 'P'):
            if any(isinstance(step, dict) and step.get('type') == 'ToTensor' and k in step.get('keys', []) for step in vp):
                keys_in_collect.append(k)
        vp.append(dict(type='Collect', keys=keys_in_collect))

    mean, std, log_base, offset = _find_label_norm_and_log(vp)
    print(f'Label StdConfig -> mean={mean}, std={std}, log_base={log_base}, offset={offset}')

    test_labels_path = 'datasetone3/test/testlabels.xlsx'
    test_labels = load_labels(test_labels_path)
    test_dataset = MyDataset(test_labels, image_dir='datasetone3/test/', cfg=vp)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=data_cfg.get('batch_size', 4),
                             num_workers=data_cfg.get('num_workers', 2),
                             pin_memory=True, drop_last=False, collate_fn=collate)

    pred_list, target_list = [], []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Test') as pbar:
            for it, batch in enumerate(test_loader):
                if it == 0:
                    print('Batch keys:', list(batch.keys()))
                    for key in ('T','V','P'):
                        if key in batch:
                            b = batch[key].float()
                            print(f'{key} mean/std: {b.mean().item():.3f}/{b.std().item():.3f}')
                imgs = batch['img'].to(device, dtype=torch.float32)
                feed = {}
                for key in ('T','V','P'):
                    if key in batch:
                        feed[key] = batch[key].to(device, dtype=torch.float32)
                preds = model(imgs, return_loss=False, **feed)
                preds_np = preds.detach().cpu().float().view(-1).numpy() if torch.is_tensor(preds) \
                           else np.asarray(preds, dtype=np.float32).reshape(-1)
                pred_list.extend(preds_np.tolist())

                if 'k' in batch:
                    k = batch['k']
                    tgt_np = k.detach().cpu().float().view(-1).numpy() if torch.is_tensor(k) \
                             else np.asarray(k, dtype=np.float32).reshape(-1)
                    target_list.extend(tgt_np.tolist())
                else:
                    target_list.extend([np.nan]*len(preds_np))
                pbar.update(1)

    # 转 tensor、过滤 NaN
    preds_std = torch.from_numpy(np.asarray(pred_list, dtype=np.float32).reshape(-1))
    tgts_std = torch.from_numpy(np.asarray(target_list, dtype=np.float32).reshape(-1))
    mask = ~torch.isnan(tgts_std)
    preds_std = preds_std[mask]; tgts_std = tgts_std[mask]

    # 标准化空间指标
    std_results = evaluate(preds_std, tgts_std,
                           data_cfg.get('test',{}).get('metrics',['rmse','r2']),
                           data_cfg.get('test',{}).get('metric_options',{}))
    print(AsciiTable([['Metric','Value']]+[[k,f'{float(v):.6f}'] for k,v in std_results.items()],
                     'Standardized Space').table)

    # 原量纲指标
    if mean is not None and std is not None:
        preds_org = _inverse_transform(preds_std, mean, std, log_base, offset)
        tgts_org = _inverse_transform(tgts_std, mean, std, log_base, offset)
        org_results = evaluate(preds_org, tgts_org,
                               data_cfg.get('test',{}).get('metrics',['rmse','r2']),
                               data_cfg.get('test',{}).get('metric_options',{}))
        print(AsciiTable([['Metric','Value']]+[[k,f'{float(v):.6f}'] for k,v in org_results.items()],
                         'Original Scale').table)

if __name__ == '__main__':
    main()
