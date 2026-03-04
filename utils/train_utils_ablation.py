# utils/train_utils.py
import os
import torch
import torch.distributed as dist
import sys
import types
import importlib
import random
from tqdm import tqdm
import numpy as np
from numpy import mean
from terminaltables import AsciiTable
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from core.evaluations import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.common import get_dist_info
from utils.history import History

def init_random_seed(seed=None, device='cuda'):
    if seed is not None:
        return seed
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed
    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def file2dict(filename):
    path, file = os.path.split(filename)
    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0, abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {name: value for name, value in mod.__dict__.items()
                if not name.startswith('__')
                and not isinstance(value, types.ModuleType)
                and not isinstance(value, types.FunctionType)}
    return (cfg_dict.get('model_cfg'),
            cfg_dict.get('train_pipeline'),
            cfg_dict.get('valid_pipeline'),
            cfg_dict.get('test_pipeline'),
            cfg_dict.get('data_cfg'),
            cfg_dict.get('lr_config'),
            cfg_dict.get('optimizer_cfg'))

def print_info(cfg):
    backbone = cfg.get('backbone', {}).get('type', 'None')
    if isinstance(cfg.get('neck'), list):
        neck = ' '.join([i.get('type') for i in cfg.get('neck')])
    else:
        neck = cfg.get('neck').get('type') if cfg.get('neck') is not None else 'None'
    head = cfg.get('head').get('type') if cfg.get('head') is not None else 'None'
    loss = cfg.get('head').get('loss').get('type') if cfg.get('head').get('loss') is not None else 'None'
    table_data = [('Backbone', 'Neck', 'Head', 'Loss'), (backbone, neck, head, loss)]
    print('\n', AsciiTable(table_data, 'Model info').table, '\n')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def resume_model(model, runner, checkpoint, meta, resume_optimizer=True, map_location='default'):
    try:
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = load_checkpoint(model, checkpoint, map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = load_checkpoint(model, checkpoint)
        else:
            checkpoint = load_checkpoint(model, checkpoint, map_location=map_location)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return model, runner, meta

    runner['epoch'] = checkpoint['meta']['epoch']
    runner['iter'] = checkpoint['meta']['iter']
    runner['best_train_weight'] = checkpoint['meta']['best_train_weight']
    runner['last_weight'] = checkpoint['meta']['last_weight']
    meta = checkpoint.get('meta', meta or {})

    if 'optimizer' in checkpoint and resume_optimizer:
        if isinstance(runner['optimizer'], Optimizer):
            runner['optimizer'].load_state_dict(checkpoint['optimizer'])
        elif isinstance(runner['optimizer'], dict):
            for k in runner['optimizer'].keys():
                runner.optimizer[k].load_state_dict(checkpoint['optimizer'][k])
        else:
            raise TypeError('Optimizer should be dict or torch.optim.Optimizer')
    print(f'resumed epoch {runner["epoch"]}, iter {runner["iter"]}')
    return model, runner, meta

def _print_tvp_stats_once(batch, tag='Train'):
    try:
        for key in ('T', 'V', 'P'):
            if key in batch:
                b = batch[key].float().detach().cpu().view(-1)
                print(f'{tag} {key}: mean={b.mean().item():.3f}, std={b.std(unbiased=False).item():.3f}')
    except Exception as e:
        print(f'Warn: TVP stats print failed: {e}')


# -------- 单帧训练（已加入 4 个排查点）--------

def apply_tvp_ablation(T, V, P, mode: str = "TVP"):
    """Apply TVP ablation by zeroing out selected conditioning variables.

    Keep passing T/V/P keys to avoid order mismatch inside PhyRegHead (fixed order: T, V, P).

    Modes:
      - TVP: keep all
      - TP:  zero V
      - TV:  zero P
      - T:   zero V and P

    Assumes T/V/P are already standardized by the data pipeline; in standardized space, 0 is mean.
    """
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



# -------- 去除图像信息（TVP-only 条件实验）--------
def apply_cond_only(images: torch.Tensor, cond_only: bool = False) -> torch.Tensor:
    """If cond_only is True, zero-out the image tensor while keeping shape unchanged.

    This removes per-sample visual information and forces the model to rely on T/V/P only.
    """
    if not cond_only:
        return images
    return torch.zeros_like(images)

def _ablation_check_once(T, V, P, mode: str, epoch: int, it: int, tag: str = "Train"):
    """Print post-ablation stats once to verify the ablation is actually applied.

    This must be called AFTER apply_tvp_ablation and BEFORE model forward.
    """
    if epoch != 0 or it != 0:
        return
    def _stat(x):
        if x is None:
            return "None"
        x = x.detach()
        return (f"mean={x.mean().item():.6f}, std={x.std(unbiased=False).item():.6f}, "
                f"min={x.min().item():.6f}, max={x.max().item():.6f}")
    print(f"[ABLATION CHECK][{tag}] mode={mode}")
    print(f"[ABLATION CHECK][{tag}] T -> {_stat(T)}")
    print(f"[ABLATION CHECK][{tag}] V -> {_stat(V)}")
    print(f"[ABLATION CHECK][{tag}] P -> {_stat(P)}")


def train(model, runner, lr_update_func, device, epoch, epoches, test_cfg, meta):
    train_loss = 0
    pred_list, target_list = [], []
    runner['epoch'] = epoch + 1
    meta['epoch'] = runner['epoch']
    model.train()
    train_loader = runner.get('train_loader')
    optimizer = runner.get('optimizer')
    acc_steps = int(runner.get('accumulate_steps', 1))
    optimizer.zero_grad(set_to_none=True)

    # === Debug 开关：只在第 1 个 epoch 打印前几个 batch ===
    DEBUG = (epoch == 0)
    DEBUG_STEPS = 3

    with tqdm(total=len(train_loader), desc=f'Train: Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
        for iter, batch in enumerate(train_loader):
            if iter == 0:
                _print_tvp_stats_once(batch, tag='Train')

            # ========= 排查 A：图像张量范围 =========
            if DEBUG and iter < DEBUG_STEPS:
                imgs_cpu = batch['img']  # 还在 CPU 上的 tensor
                try:
                    print(f"\n[DEBUG][Epoch {epoch} Batch {iter}] img stats -> "
                          f"min={imgs_cpu.min().item():.4f}, "
                          f"max={imgs_cpu.max().item():.4f}, "
                          f"mean={imgs_cpu.mean().item():.4f}, "
                          f"std={imgs_cpu.std().item():.4f}")
                except Exception as e:
                    print(f"[DEBUG] img stats print failed: {e}")

                # ========= 排查 B：文件名是否正常 / 多样 =========
                if 'filename' in batch:
                    print(f"[DEBUG][Epoch {epoch} Batch {iter}] filenames sample -> {batch['filename'][:5]}")
                else:
                    print(f"[DEBUG][Epoch {epoch} Batch {iter}] WARNING: 'filename' not in batch keys: {list(batch.keys())}")

            images = batch['img'].to(device, dtype=torch.float32)
            images = apply_cond_only(images, runner.get('cond_only', False))
            targets = batch['k'].to(device, dtype=torch.float32).unsqueeze(1)

            # ========= 排查 C：log+标准化后的 k 分布 =========
            if DEBUG and iter < DEBUG_STEPS:
                try:
                    t_cpu = targets.detach().cpu()
                    print(f"[DEBUG][Epoch {epoch} Batch {iter}] k (after log+std) stats -> "
                          f"mean={t_cpu.mean().item():.4f}, std={t_cpu.std().item():.4f}")
                except Exception as e:
                    print(f"[DEBUG] k stats print failed: {e}")

            T = batch.get('T', None); V = batch.get('V', None); P = batch.get('P', None)
            if T is not None: T = T.to(device, dtype=torch.float32)
            if V is not None: V = V.to(device, dtype=torch.float32)
            if P is not None: P = P.to(device, dtype=torch.float32)

            T, V, P = apply_tvp_ablation(T, V, P, runner.get('tvp_mode', 'TVP'))

            _ablation_check_once(T, V, P, runner.get('tvp_mode','TVP'), epoch, iter, tag='Train')
            lr_update_func.before_train_iter(runner)
            preds, losses = model(images, targets=targets, T=T, V=V, P=P, return_loss=True, train_statu=True)
            loss = losses['loss']
            train_loss += loss.item()
            loss = loss / acc_steps
            loss.backward()

            # ========= 排查 D：梯度是否正常 =========
            if DEBUG and iter < DEBUG_STEPS:
                grad_ok = False
                try:
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            gmean = param.grad.abs().mean().item()
                            print(f"[DEBUG][Epoch {epoch} Batch {iter}] grad mean of '{name}' -> {gmean:.6e}")
                            grad_ok = True
                            break
                    if not grad_ok:
                        print(f"[DEBUG][Epoch {epoch} Batch {iter}] WARNING: no parameter has grad "
                              f"(可能是模型被冻结/未正确 backward/输入异常)")
                except Exception as e:
                    print(f"[DEBUG] grad check failed: {e}")

            if ((iter + 1) % acc_steps == 0) or (iter + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pred_list.append(preds.detach()); target_list.append(targets.detach())
            pbar.set_postfix(Loss=train_loss / (iter + 1), Lr=get_lr(optimizer))
            runner['iter'] += 1; meta['iter'] = runner['iter']
            pbar.update(1)

    meta['predicted_values'] = torch.cat(pred_list)
    meta['true_values'] = torch.cat(target_list)
    eval_results = evaluate(meta['predicted_values'], meta['true_values'], test_cfg['metrics'], test_cfg['metric_options'])
    meta['train_info']['train_loss'].append(train_loss / (iter + 1))
    meta['train_info']['train_metric'].append(eval_results)

    if train_loss / len(runner.get('train_loader')) < runner.get('best_train_loss'):
        runner['best_train_loss'] = train_loss / len(runner.get('train_loader'))
        meta['best_train_loss'] = runner['best_train_loss']
        if epoch > 0 and os.path.isfile(runner['best_train_weight']):
            os.remove(runner['best_train_weight'])
        runner['best_train_weight'] = os.path.join(meta['save_dir'], f'Train_Epoch{epoch+1:03}-Loss{runner["best_train_loss"]:.3f}.pth')
        meta['best_train_weight'] = runner['best_train_weight']
        save_checkpoint(model, runner['best_train_weight'], optimizer, meta)

    table_data = [('Root Mean Square Error', 'R2_Score'),
                  ('{:.4f}'.format(mean(eval_results.get('rmse', 0.0))),
                   '{:.4f}'.format(mean(eval_results.get('r2', 0.0)))) ]
    print('\n', AsciiTable(table_data, 'Train Results').table, '\n')

def validation(model, runner, cfg, device, epoch, epoches, meta):
    pred_list, target_list = [], []
    val_loss = 0.0
    model.eval()
    val_loader = runner.get('val_loader')
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Valid : Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
            for iter, batch in enumerate(val_loader):
                if iter == 0:
                    _print_tvp_stats_once(batch, tag='Valid')
                images = batch['img'].to(device, dtype=torch.float32)
                images = apply_cond_only(images, runner.get('cond_only', False))
                targets = batch['k'].to(device, dtype=torch.float32).unsqueeze(1)
                T = batch.get('T', None); V = batch.get('V', None); P = batch.get('P', None)
                if T is not None: T = T.to(device, dtype=torch.float32)
                if V is not None: V = V.to(device, dtype=torch.float32)
                if P is not None: P = P.to(device, dtype=torch.float32)
                T, V, P = apply_tvp_ablation(T, V, P, runner.get('tvp_mode', 'TVP'))
                _ablation_check_once(T, V, P, runner.get('tvp_mode','TVP'), epoch, iter, tag='Valid')
                preds, losses = model(images, targets=targets, T=T, V=V, P=P, return_loss=True, train_statu=True)
                loss = losses['loss']
                pred_list.append(preds.detach()); target_list.append(targets.detach())
                val_loss += loss.item()
                pbar.set_postfix(Loss=val_loss / (iter + 1)); pbar.update(1)

    meta['predicted_values'] = torch.cat(pred_list)
    meta['true_values'] = torch.cat(target_list)
    eval_results = evaluate(meta['predicted_values'], meta['true_values'], cfg.get('metrics'), cfg.get('metric_options'))
    meta['train_info']['val_metric'].append(eval_results)
    meta['train_info']['val_loss'].append(val_loss / (iter + 1))

    table_data = [('Root Mean Square Error', 'R2_Score'),
                  ('{:.4f}'.format(mean(eval_results.get('rmse', 0.0))),
                   '{:.4f}'.format(mean(eval_results.get('r2', 0.0)))) ]
    print('\n', AsciiTable(table_data, 'Validation Results').table, '\n')

    if eval_results.get('rmse') < runner.get('best_val_rmse'):
        runner['best_val_rmse'] = eval_results.get('rmse')
        meta['best_val_rmse'] = runner['best_val_rmse']
        if epoch > 0 and os.path.isfile(runner['best_val_weight']):
            os.remove(runner['best_val_weight'])
        runner['best_val_weight'] = os.path.join(meta['save_dir'], f'Val_Epoch{epoch+1:03}-RMSE{eval_results.get("rmse"):.3f}.pth')
        meta['best_val_weight'] = runner['best_val_weight']
        save_checkpoint(model, runner['best_val_weight'], runner['optimizer'], meta)

    if epoch > 0 and os.path.isfile(runner['last_weight']):
        os.remove(runner['last_weight'])
    runner['last_weight'] = os.path.join(meta['save_dir'], f'Last_Epoch{epoch + 1:03}.pth')
    meta['last_weight'] = runner['last_weight']
    save_checkpoint(model, runner['last_weight'], runner['optimizer'], meta)

def test(model, runner, cfg, device, meta):
    pred_list, target_list = [], []
    model.eval()
    test_loader = runner.get('test_loader')
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f'Test', postfix=dict, mininterval=0.3) as pbar:
            for iter, batch in enumerate(test_loader):
                images = batch['img'].to(device, dtype=torch.float32)
                images = apply_cond_only(images, runner.get('cond_only', False))
                targets = batch['k'].to(device, dtype=torch.float32).unsqueeze(1)
                T = batch.get('T', None); V = batch.get('V', None); P = batch.get('P', None)
                if T is not None: T = T.to(device, dtype=torch.float32)
                if V is not None: V = V.to(device, dtype=torch.float32)
                if P is not None: P = P.to(device, dtype=torch.float32)
                T, V, P = apply_tvp_ablation(T, V, P, runner.get('tvp_mode', 'TVP'))
                _ablation_check_once(T, V, P, runner.get('tvp_mode','TVP'), 0, iter, tag='Test')
                preds = model(images, T=T, V=V, P=P, return_loss=False, train_statu=False)
                pred_list.append(preds.detach()); target_list.append(targets.to(device)); pbar.update(1)

    meta['predicted_values'] = torch.cat(pred_list).reshape(-1, 1)
    meta['true_values'] = torch.cat(target_list).reshape(-1, 1)
    eval_results = evaluate(meta['predicted_values'], meta['true_values'], cfg['metrics'], cfg['metric_options'])
    meta['test_info']['test_metric'].append(eval_results)

    table_data = [('Root Mean Square Error', 'R2_Score'),
                  ('{:.4f}'.format(eval_results.get('rmse', 0.0)),
                   '{:.4f}'.format(eval_results.get('r2', 0.0))) ]
    print('\n', AsciiTable(table_data, 'Test Results').table, '\n')

    test_history = History(meta['save_dir'])
    scatter_path = os.path.join(meta['save_dir'], 'test_scatter_plot.png')
    err_scatter = os.path.join(meta['save_dir'], 'test_error_scatter.png')
    err_hist = os.path.join(meta['save_dir'], 'test_error_hist.png')
    test_history.draw_test_results(meta, scatter_path)
    test_history.draw_error_plots(meta, err_scatter, err_hist)
    test_history.save_test_predtrue(meta)
    test_history.save_test_metrics(meta)

# ===== Temporal training with accumulation & TVP print =====
def _flatten_pair_for_eval(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
    p = preds.float().reshape(-1, 1)
    t = targets.float().reshape(-1, 1) if targets is not None else torch.empty(0, 1)
    if mask is None or t.numel() == 0:
        return p, t
    m = mask.float().reshape(-1, 1)
    sel = (m > 0.5).view(-1)
    if sel.any():
        return p[sel], t[sel]
    else:
        return torch.empty(0, 1, dtype=p.dtype), torch.empty(0, 1, dtype=t.dtype)

def train_temporal(model, runner, lr_update_func, device, epoch, epoches, test_cfg, meta):
    train_loss = 0.0
    pred_list, target_list = [], []
    runner['epoch'] = epoch + 1
    meta['epoch'] = runner['epoch']

    model.train()
    train_loader = runner.get('train_loader')
    optimizer = runner.get('optimizer')
    acc_steps = int(runner.get('accumulate_steps', 1))
    optimizer.zero_grad(set_to_none=True)

    with tqdm(total=len(train_loader), desc=f'Train(Temporal): Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
        for it, batch in enumerate(train_loader):
            if it == 0:
                _print_tvp_stats_once(batch, tag='Train(T)')
            images = batch['img'].to(device, dtype=torch.float32)  # [B,T,C,H,W]
            images = apply_cond_only(images, runner.get('cond_only', False))
            targets = batch.get('k', None)
            k_mask  = batch.get('k_mask', None)
            stage = batch.get('stage', None)
            T = batch.get('T', None); V = batch.get('V', None); P = batch.get('P', None)

            if targets is not None: targets = targets.to(device, dtype=torch.float32)
            if k_mask  is not None: k_mask  = k_mask.to(device, dtype=torch.float32)
            if stage   is not None: stage   = stage.to(device, dtype=torch.long)
            if T is not None: T = T.to(device, dtype=torch.float32)
            if V is not None: V = V.to(device, dtype=torch.float32)
            if P is not None: P = P.to(device, dtype=torch.float32)

            T, V, P = apply_tvp_ablation(T, V, P, runner.get('tvp_mode', 'TVP'))

            _ablation_check_once(T, V, P, runner.get('tvp_mode','TVP'), epoch, it, tag='Train(T)')
            lr_update_func.before_train_iter(runner)
            preds, losses = model(images, targets=targets, k_mask=k_mask, stage=stage, T=T, V=V, P=P,
                                  return_loss=True, train_statu=True)
            loss = losses['loss']
            train_loss += loss.item()
            (loss / acc_steps).backward()

            if ((it + 1) % acc_steps == 0) or (it + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if targets is not None and k_mask is not None:
                p_flat, t_flat = _flatten_pair_for_eval(preds.detach(), targets.detach(), k_mask.detach())
                if t_flat.numel() > 0:
                    pred_list.append(p_flat)
                    target_list.append(t_flat)

            pbar.set_postfix(Loss=train_loss / (it + 1), Lr=get_lr(optimizer))
            runner['iter'] += 1; meta['iter'] = runner['iter']
            pbar.update(1)

    if len(pred_list) > 0:
        meta['predicted_values'] = torch.cat(pred_list, dim=0)
        meta['true_values'] = torch.cat(target_list, dim=0)
        eval_results = evaluate(meta['predicted_values'], meta['true_values'],
                                test_cfg['metrics'], test_cfg['metric_options'])
    else:
        eval_results = {'rmse': float('nan'), 'r2': float('nan')}

    meta['train_info']['train_loss'].append(train_loss / (it + 1))
    meta['train_info']['train_metric'].append(eval_results)

    if train_loss / len(train_loader) < runner.get('best_train_loss'):
        runner['best_train_loss'] = train_loss / len(train_loader)
        meta['best_train_loss'] = runner['best_train_loss']
        if epoch > 0 and os.path.isfile(runner['best_train_weight']):
            os.remove(runner['best_train_weight'])
        runner['best_train_weight'] = os.path.join(meta['save_dir'], f'Train_Epoch{epoch+1:03}-Loss{runner["best_train_loss"]:.3f}.pth')
        meta['best_train_weight'] = runner['best_train_weight']
        save_checkpoint(model, runner['best_train_weight'], optimizer, meta)

    table = [('Root Mean Square Error', 'R2_Score'),
             ('{:.4f}'.format(np.nanmean(eval_results.get('rmse', 0.0))),
              '{:.4f}'.format(np.nanmean(eval_results.get('r2', 0.0)))) ]
    print('\n', AsciiTable(table, 'Train Temporal Results').table, '\n')

def validation_temporal(model, runner, cfg, device, epoch, epoches, meta):
    pred_list, target_list = [], []
    val_loss = 0.0
    model.eval()
    val_loader = runner.get('val_loader')
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Valid(Temporal): Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
            for it, batch in enumerate(val_loader):
                if it == 0:
                    _print_tvp_stats_once(batch, tag='Valid(T)')
                images = batch['img'].to(device, dtype=torch.float32)
                images = apply_cond_only(images, runner.get('cond_only', False))
                targets = batch.get('k', None)
                k_mask  = batch.get('k_mask', None)
                stage = batch.get('stage', None)
                T = batch.get('T', None); V = batch.get('V', None); P = batch.get('P', None)
                if targets is not None: targets = targets.to(device, dtype=torch.float32)
                if k_mask  is not None: k_mask  = k_mask.to(device, dtype=torch.float32)
                if stage   is not None: stage   = stage.to(device, dtype=torch.long)
                if T is not None: T = T.to(device, dtype=torch.float32)
                if V is not None: V = V.to(device, dtype=torch.float32)
                if P is not None: P = P.to(device, dtype=torch.float32)

                T, V, P = apply_tvp_ablation(T, V, P, runner.get('tvp_mode', 'TVP'))

                _ablation_check_once(T, V, P, runner.get('tvp_mode','TVP'), epoch, it, tag='Valid(T)')
                preds, losses = model(images, targets=targets, k_mask=k_mask, stage=stage, T=T, V=V, P=P,
                                      return_loss=True, train_statu=True)
                loss = losses['loss']; val_loss += loss.item()

                if targets is not None and k_mask is not None:
                    p_flat, t_flat = _flatten_pair_for_eval(preds.detach(), targets.detach(), k_mask.detach())
                    if t_flat.numel() > 0:
                        pred_list.append(p_flat)
                        target_list.append(t_flat)
                pbar.set_postfix(Loss=val_loss / (it + 1)); pbar.update(1)

    if len(pred_list) > 0:
        meta['predicted_values'] = torch.cat(pred_list, dim=0)
        meta['true_values'] = torch.cat(target_list, dim=0)
        eval_results = evaluate(meta['predicted_values'], meta['true_values'],
                                cfg.get('metrics'), cfg.get('metric_options'))
    else:
        eval_results = {'rmse': float('nan'), 'r2': float('nan')}

    meta['train_info']['val_metric'].append(eval_results)
    meta['train_info']['val_loss'].append(val_loss / (it + 1))

    table = [('Root Mean Square Error', 'R2_Score'),
             ('{:.4f}'.format(np.nanmean(eval_results.get('rmse', 0.0))),
              '{:.4f}'.format(np.nanmean(eval_results.get('r2', 0.0)))) ]
    print('\n', AsciiTable(table, 'Validation Temporal Results').table, '\n')

    if eval_results.get('rmse', float('inf')) < runner.get('best_val_rmse'):
        runner['best_val_rmse'] = eval_results.get('rmse')
        meta['best_val_rmse'] = runner['best_val_rmse']
        if epoch > 0 and os.path.isfile(runner['best_val_weight']):
            os.remove(runner['best_val_weight'])
        runner['best_val_weight'] = os.path.join(meta['save_dir'], f'Val_Epoch{epoch+1:03}-RMSE{runner["best_val_rmse"]:.3f}.pth')
        meta['best_val_weight'] = runner['best_val_weight']
        save_checkpoint(model, runner['best_val_weight'], runner['optimizer'], meta)

    if epoch > 0 and os.path.isfile(runner['last_weight']):
        os.remove(runner['last_weight'])
    runner['last_weight'] = os.path.join(meta['save_dir'], f'Last_Epoch{epoch + 1:03}.pth')
    meta['last_weight'] = runner['last_weight']
    save_checkpoint(model, runner['last_weight'], runner['optimizer'], meta)

def test_temporal(model, runner, cfg, device, meta):
    pred_list, target_list = [], []
    model.eval()
    test_loader = runner.get('test_loader')

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Test(Temporal)', postfix=dict, mininterval=0.3) as pbar:
            for it, batch in enumerate(test_loader):
                images = batch['img'].to(device, dtype=torch.float32)  # [B,T,C,H,W]
                images = apply_cond_only(images, runner.get('cond_only', False))
                targets = batch.get('k', None)
                k_mask  = batch.get('k_mask', None)
                T = batch.get('T', None); V = batch.get('V', None); P = batch.get('P', None)
                if targets is not None: targets = targets.to(device, dtype=torch.float32)
                if k_mask  is not None: k_mask  = k_mask.to(device, dtype=torch.float32)
                if T is not None: T = T.to(device, dtype=torch.float32)
                if V is not None: V = V.to(device, dtype=torch.float32)
                if P is not None: P = P.to(device, dtype=torch.float32)

                T, V, P = apply_tvp_ablation(T, V, P, runner.get('tvp_mode', 'TVP'))

                _ablation_check_once(T, V, P, runner.get('tvp_mode','TVP'), 0, it, tag='Test(T)')
                preds = model(images, T=T, V=V, P=P, return_loss=False, train_statu=False)
                p = preds.float().reshape(-1, 1).detach().cpu()
                if targets is not None and k_mask is not None:
                    t = targets.float().reshape(-1, 1).detach().cpu()
                    m = k_mask.float().reshape(-1, 1).detach().cpu()
                    sel = (m > 0.5).view(-1)
                    if sel.any():
                        pred_list.append(p[sel]); target_list.append(t[sel])
                pbar.update(1)

    if len(pred_list) == 0:
        print('No temporal predictions with labels to evaluate.')
        return

    meta['predicted_values'] = torch.cat(pred_list, dim=0)
    meta['true_values'] = torch.cat(target_list, dim=0)
    eval_results = evaluate(meta['predicted_values'], meta['true_values'], cfg['metrics'], cfg['metric_options'])
    meta['test_info']['test_metric'].append(eval_results)

    table_data = [('Root Mean Square Error', 'R2_Score'),
                  ('{:.4f}'.format(eval_results.get('rmse', 0.0)),
                   '{:.4f}'.format(eval_results.get('r2', 0.0)))]
    print('\n', AsciiTable(table_data, 'Test Temporal Results').table, '\n')

    test_history = History(meta['save_dir'])
    scatter_path = os.path.join(meta['save_dir'], 'test_scatter_plot.png')
    err_scatter = os.path.join(meta['save_dir'], 'test_error_scatter.png')
    err_hist = os.path.join(meta['save_dir'], 'test_error_hist.png')
    test_history.draw_test_results(meta, scatter_path)
    test_history.draw_error_plots(meta, err_scatter, err_hist)
    test_history.save_test_predtrue(meta)
    test_history.save_test_metrics(meta)