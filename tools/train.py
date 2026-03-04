import json
import os
import sys
sys.path.insert(0, os.getcwd())
import copy
import argparse
import shutil
import time
import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch.optim as optim
from utils.history import History
from utils.dataloader import MyDataset, collate
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model, \
    test
from utils.inference import init_model
from models.build import BuildNet
from core.optimizers.lr_update import StepLrUpdater, LrUpdater, PolyLrUpdater, CosineAnnealingLrUpdater, \
    CosineAnnealingCooldownLrUpdater, ReduceLROnPlateauLrUpdater
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--split-validation',
        action='store_true',
        help='whether to split validation set from training set.')
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='the proportion of the validation set to the training set.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def load_labels(file_path):
    try:
        # 读取Excel文件
        dataframe = pd.read_excel(file_path)
        return dataframe
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {file_path}, {e}")
        return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 读取配置文件获取关键字段
    args = parse_args()
    model_cfg, train_pipeline, valid_pipeline, test_pipeline,data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    print_info(model_cfg)
    # 初始化
    meta = dict()
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('logs', model_cfg.get('backbone').get('type'), dirname)
    meta['save_dir'] = save_dir

    # 设置随机数种子
    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=args.deterministic)
    meta['seed'] = seed

    # 读取训练标签数据
    train_labels_path = 'datasetone3/train/trainlabels.xlsx'
    train_labels = load_labels(train_labels_path)

    # 读取验证标签数据
    val_labels_path = 'datasetone3/valid/validlabels.xlsx'
    val_labels = load_labels(val_labels_path)
    # 读取测试标签数据
    test_labels_path = 'datasetone3/test/testlabels.xlsx'
    test_labels = load_labels(test_labels_path)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        # 检查是否有可用的 GPU
        if torch.cuda.is_available():
            # 设置默认设备为 GPU
            device = torch.device("cuda")
            print('GPU found: {}'.format(torch.cuda.get_device_name(0)))  # 0 表示第一个 GPU 设备
        else:
            # 如果没有 GPU，使用 CPU
            device = torch.device("cpu")
            print('No GPU found.')

    print('Initialize the weights.')

    model = BuildNet(model_cfg)

    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()
    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        freeze_layers = ' '.join(list(data_cfg.get('train').get('freeze_layers')))
        print('Freeze layers : ' + freeze_layers)
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))

    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])

    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")
    meta['total_params'] = total_params
    # 初始化优化器
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)

    # 初始化学习率更新策略
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)
    train_dataset = MyDataset(train_labels, image_dir='datasetone3/train/', cfg=train_pipeline)
    val_dataset = MyDataset(val_labels, image_dir='datasetone3/valid/', cfg=valid_pipeline)
    test_dataset = MyDataset(test_labels, image_dir='datasetone3/test/', cfg=test_pipeline)


    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'),
                              num_workers=data_cfg.get('num_workers'), pin_memory=True, drop_last=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=data_cfg.get('batch_size'),
                            num_workers=data_cfg.get('num_workers'), pin_memory=True,
                            drop_last=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=data_cfg.get('batch_size'),
                            num_workers=data_cfg.get('num_workers'), pin_memory=True,
                            drop_last=False, collate_fn=collate)

    # 将关键字段存储，方便训练时同步调用&更新
    runner = dict(
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        iter=0,
        epoch=0,
        max_epochs=data_cfg.get('train').get('epoches'),
        max_iters=data_cfg.get('train').get('epoches') * len(train_loader),
        best_train_loss=float('INF'),
        best_val_rmse=float('INF'),
        best_train_weight='',
        best_val_weight='',
        last_weight=''
    )
    meta['train_info'] = dict(train_loss=[],
                              val_loss=[],
                              train_metric=[],
                              val_metric=[])
    meta['test_info'] = dict(test_loss= [],
                             test_metric= []
    )
    # 是否从中断处恢复训练
    if args.resume_from:
        model, runner, meta = resume_model(model, runner, args.resume_from, meta)
    else:
        os.makedirs(save_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(save_dir, os.path.split(args.config)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')

    # 初始化保存训练信息类
    train_history = History(meta['save_dir'])

    # 记录初始学习率，
    lr_update_func.before_run(runner)
    start_time = time.time()

    # 训练
    for epoch in range(runner.get('epoch'), runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)
        train(model, runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'),data_cfg.get('test'),meta)
        validation(model, runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)
        torch.cuda.empty_cache()
        train_history.after_epoch(meta)


    test(model, runner, data_cfg.get('test'), device, meta)






if __name__ == "__main__":
    main()
