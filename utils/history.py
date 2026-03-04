import matplotlib
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
matplotlib.use('Agg')
import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from numpy import mean

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from numpy import VisibleDeprecationWarning
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    warnings.filterwarnings("ignore")  # 忽略所有警告，或写你需要的类别
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class History:
    def __init__(self, dir):
        self.dir = dir
        self.csv_dir = os.path.join(dir, 'metrics_outputs.csv')
        self.test_csv_dir = os.path.join(dir, 'test_results.csv')
        self.test_predtrue_path = os.path.join(dir, 'test_predtrue.csv')
        self.pic_dir = os.path.join(dir, 'metric-epoch.png')
        self.losses_epoch = []
        self.epoch_outputs = [['Epoch', 'Train Loss', 'Val Metric']]
        # self.temp_data = []
        self.test_results = []

    def draw_loss_epoch(self, train_loss, val_loss,save_path):
        total_epoch = range(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(total_epoch, train_loss, 'red', linewidth=2, label='Training')
        plt.plot(total_epoch, val_loss, 'blue', linewidth=2, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close("all")


    def draw_r2_epoch(self, train_r2, val_r2,save_path):
        total_epoch = range(1, len(train_r2) + 1)

        plt.figure()
        plt.plot(total_epoch, train_r2, 'red', linewidth=2, label='Training')
        plt.plot(total_epoch, val_r2, 'blue', linewidth=2, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('R2')
        # plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close("all")



    def after_epoch(self, meta):
        '''
        保存每周期的 'Train RMSE' 和 'Val RMSE'
           meta['train_info'] = dict(train_loss = [],
                              val_loss = [],
                              train_metric=[],
                              val_metric=[])
        '''

        val_rmse_epoch = []
        train_rmse_epoch = []
        val_r2_epoch = []
        train_r2_epoch = []


        epoch_outputs = [
            ['index', 'train_loss', 'train_rmse', 'train_r2','val_loss', 'val_rmse', ]]
        with open(self.csv_dir, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(meta['train_info']['train_loss'])):
                temp_data = [i + 1, mean(meta['train_info']['train_loss'][i]),
                             mean(meta['train_info']['train_metric'][i].get('rmse')),
                             mean(meta['train_info']['train_metric'][i].get('r2')),
                             mean(meta['train_info']['val_loss'][i]),
                             mean(meta['train_info']['val_metric'][i].get('rmse')),
                             mean(meta['train_info']['val_metric'][i].get('r2')),
                             ]

                val_rmse_epoch.append(meta['train_info']['val_metric'][i].get('rmse'))
                train_rmse_epoch.append(meta['train_info']['train_metric'][i].get('rmse'))

                val_r2_epoch.append(meta['train_info']['val_metric'][i].get('r2'))
                train_r2_epoch.append(meta['train_info']['train_metric'][i].get('r2'))


                epoch_outputs.append(temp_data)
                # print(f"Epoch {i + 1} data: {temp_data}")
            writer.writerows(epoch_outputs)

        '''
        绘制每周期 Train 和 Val 的 RMSE
        '''

        r2_pic_path = os.path.join(self.dir, 'r2-epoch.png')
        self.draw_r2_epoch(train_r2_epoch, val_r2_epoch, r2_pic_path)


        loss_pic_path = os.path.join(self.dir, 'loss-epoch.png')
        self.draw_loss_epoch(meta['train_info']['train_loss'], meta['train_info']['val_loss'], loss_pic_path)


    def save_val_predtrue(self, meta, save_path=None):  # 重命名方法
        best_epoch_true_values = meta['best_val_epoch_true_values'].cpu().detach().numpy()
        best_epoch_predicted_values = meta['best_val_epoch_predicted_values'].cpu().detach().numpy()

        # 将结果保存到CSV文件
        results_df = pd.DataFrame({
            'True Values': best_epoch_true_values,
            'Predicted Values': best_epoch_predicted_values
        })
        # 使用 self.test_predtrue_csv_path 作为默认保存路径
        if save_path is None:
            save_path = self.test_predtrue_path

        results_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    def save_test_predtrue(self, meta, save_path=None):  # 重命名方法
        true_values = meta['true_values'].cpu().numpy().flatten()
        predicted_values = meta['predicted_values'].cpu().numpy().flatten()
        # # 逆变换回原始数据

        # 将结果保存到CSV文件
        results_df = pd.DataFrame({
            'True Values': true_values,
            'Predicted Values': predicted_values
        })
        # 使用 self.test_predtrue_csv_path 作为默认保存路径
        if save_path is None:
            save_path = self.test_predtrue_path

        results_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    def draw_test_results(self, meta, save_path):
        true_values = meta['true_values'].cpu().numpy()
        predicted_values = meta['predicted_values'].cpu().numpy()
        # 逆变换回原始数据



        plt.scatter(true_values, predicted_values, color='blue', label='Predicted Values', edgecolors='black')
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'k--', lw=4)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close('all')

    def draw_error_plots(self, meta, scatter_plot_path, hist_plot_path):
        # 取出并统一到 1D float
        true_values = meta['true_values']
        pred_values = meta['predicted_values']

        # 兼容 tensor / numpy
        if torch.is_tensor(true_values):
            true_values = true_values.detach().cpu().numpy()
        if torch.is_tensor(pred_values):
            pred_values = pred_values.detach().cpu().numpy()

        # 压平成 1D 并确保为 float32
        true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
        pred_values = np.asarray(pred_values, dtype=np.float32).reshape(-1)

        # 对齐长度（防御）
        n = min(true_values.shape[0], pred_values.shape[0])
        true_values = true_values[:n]
        pred_values = pred_values[:n]

        # 计算绝对误差（1D）
        absolute_error = pred_values - true_values

        # 图1：绝对误差散点（含竖线）
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=2)
        ax.scatter(true_values, absolute_error, color='blue', label='Absolute Error',
                   alpha=0.6, s=12, edgecolors='none')
        # 逐点竖线，用标量避免 inhomogeneous shape
        for xv, yv in zip(true_values, absolute_error):
            xv = float(xv);
            yv = float(yv)
            ax.plot([xv, xv], [0.0, yv], color='blue', linestyle='-', linewidth=1.2, alpha=0.5)

        # 可选坐标范围（按你之前固定的范围也可）
        # ax.set_xlim([0, 4000])
        # ax.set_ylim([-800, 800])

        # 相对误差（带 0 保护，清理 NaN/Inf）
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = np.divide(absolute_error, true_values, where=(true_values != 0))
        relative_error = np.where(np.isfinite(relative_error), relative_error, 0.0)

        ax2 = ax.twinx()
        ax2.scatter(true_values, relative_error, color='magenta', label='Relative Error',
                    alpha=0.4, s=10, edgecolors='none')
        for xv, yv in zip(true_values, relative_error):
            xv = float(xv);
            yv = float(yv)
            ax2.plot([xv, xv], [0.0, yv], color='magenta', linestyle='-', linewidth=1.0, alpha=0.4)

        # 可选限制相对误差范围
        # ax2.set_ylim([-1, 1])

        # 图例
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.91, 0.93), ncol=1)

        ax.set_xlabel('Permeability (mD)')
        ax.set_ylabel('Absolute Error (mD)', color='blue')
        ax2.set_ylabel('Relative Error', color='magenta')
        plt.title('Absolute Error and Relative Error')

        plt.tight_layout()
        plt.savefig(scatter_plot_path)
        plt.close(fig)

        # 图2：绝对误差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(absolute_error, bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.xlabel('Absolute Error (mD)')
        plt.ylabel('Frequency')
        plt.title('Absolute Error Distribution')
        plt.tight_layout()
        plt.savefig(hist_plot_path)
        plt.close()

    def save_test_metrics(self,meta):
        if 'test_info' not in meta or not meta['test_info'].get('test_metric'):
            print("No test results to save.")
            return
        with open(self.test_csv_dir, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['RMSE',  'R2_Score'])
            for test_metrics in meta['test_info']['test_metric']:
                row = [
                    mean(test_metrics.get('rmse', 0.0)),
                    mean(test_metrics.get('r2', 0.0)),
                ]
                writer.writerow(row)

        # print(f"Test metrics saved to {self.test_csv_dir}")






























