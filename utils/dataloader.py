import copy
import numpy as np
from torch.utils.data import Dataset
import torch
from core.datasets.compose import Compose

class MyDataset(Dataset):
    def __init__(self, dataframe, image_dir, cfg):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # 走完整 pipeline，返回 dict
        results = self.pipeline(copy.deepcopy(self.data_infos[index]))
        # 关键键检查
        if 'img' not in results:
            raise KeyError("'img' not found in results after pipeline")
        # filename 可能在 LoadImageFromFile 或 Collect 中被加入
        if 'filename' not in results:
            # 尝试从 img_info 兜底
            if 'img_info' in results and 'filename' in results['img_info']:
                results['filename'] = results['img_info']['filename']
            else:
                results['filename'] = ''
        return results

    def load_annotations(self):
        data_infos = []
        # dataframe 可能包含 index(或 filename)、k、T、V、P
        has_T = 'T' in self.dataframe.columns
        has_V = 'V' in self.dataframe.columns
        has_P = 'P' in self.dataframe.columns
        has_filename = 'filename' in self.dataframe.columns

        for _, row in self.dataframe.iterrows():
            img_prefix = self.image_dir
            if has_filename:
                filename = str(row['filename'])
            else:
                # 默认以 index 命名
                filename = f"{int(row['index'])}.jpg"

            k = row['k']
            info = {
                'img_prefix': img_prefix,
                'img_info': {'filename': filename},
                'k': np.array(k, dtype=np.float32),
            }
            if has_T:
                info['T'] = np.array(row['T'], dtype=np.float32)
            if has_V:
                info['V'] = np.array(row['V'], dtype=np.float32)
            if has_P:
                info['P'] = np.array(row['P'], dtype=np.float32)

            data_infos.append(info)

        return data_infos

def collate(batches):
    """将一批 results 字典合并为 batch 字典。"""
    # batches: List[dict]
    # 必要键
    imgs = [b['img'] for b in batches]
    ks = [b['k'] for b in batches]
    fns = [b.get('filename', '') for b in batches]

    # 堆叠为张量
    if torch.is_tensor(imgs[0]):
        images = torch.stack(imgs, dim=0)
    else:
        images = torch.stack([torch.as_tensor(x) for x in imgs], dim=0)

    if torch.is_tensor(ks[0]):
        k = torch.stack([x.view(-1)[0] if x.numel() == 1 else x for x in ks], dim=0).float()
    else:
        k = torch.as_tensor(ks, dtype=torch.float32)

    batch = {'img': images, 'k': k, 'filename': fns}

    # 可选键：T/V/P
    for key in ('T', 'V', 'P'):
        if key in batches[0]:
            vals = [b[key] for b in batches]
            if torch.is_tensor(vals[0]):
                t = torch.stack([v.view(-1)[0] if v.numel() == 1 else v for v in vals], dim=0).float()
            else:
                t = torch.as_tensor(vals, dtype=torch.float32)
            batch[key] = t

    return batch
