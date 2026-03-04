# utils/dataloader_seq.py
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from core.datasets.compose import Compose

class SequenceDataset(Dataset):
    """构造定长时序样本（滑窗 stride=1）
    需要列：seq_id, t, filename或index, k(可选), stage(可选), T/V/P(可选)
    支持稀疏监督：k 为空/NaN 的时刻将生成 k_mask=0
    """
    def __init__(self, dataframe, image_dir, pipeline_cfg, seq_len=8):
        self.df = dataframe.copy()
        self.image_dir = image_dir
        self.seq_len = int(seq_len)
        _pipe = [st for st in pipeline_cfg if not (isinstance(st, dict) and st.get('type') == 'Collect')]
        self.pipeline = Compose(_pipe)

        # 序列内按 t 排序
        if 'seq_id' in self.df.columns and 't' in self.df.columns:
            self.df.sort_values(by=['seq_id', 't'], inplace=True)

        # 构造滑窗索引
        groups = self.df.groupby('seq_id') if 'seq_id' in self.df.columns else [(0, self.df)]
        self.seqs = []
        for _, g in groups:
            idx = g.index.tolist()
            if len(idx) < self.seq_len:
                continue
            for i in range(0, len(idx) - self.seq_len + 1):
                self.seqs.append(idx[i:i + self.seq_len])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        idxs = self.seqs[i]
        frames = []
        for ridx in idxs:
            row = self.df.loc[ridx]
            # filename 优先，其次 index
            if 'filename' in self.df.columns and isinstance(row.get('filename', ''), str) and row['filename']:
                fname = str(row['filename'])
            else:
                if 'index' not in self.df.columns:
                    raise KeyError("SequenceDataset需要 'filename' 或 'index' 列")
                fname = f"{int(row['index'])}.jpg"

            data = dict(img_prefix=self.image_dir, img_info=dict(filename=fname))
            # 附带条件与标签（稀疏 k 可为空）
            for k in ('k', 'T', 'V', 'P', 'stage'):
                if k in row and (not (isinstance(row[k], float) and np.isnan(row[k]))):
                    data[k] = np.array(row[k], dtype=np.float32)

            res = self.pipeline(copy.deepcopy(data))
            if 'filename' not in res:
                res['filename'] = res.get('img_info', {}).get('filename', '')
            frames.append(res)

        # 堆叠图像 [T,C,H,W]
        imgs = torch.stack([f['img'] if torch.is_tensor(f['img']) else torch.as_tensor(f['img'])
                            for f in frames], dim=0)
        pack = dict(img=imgs)

        def _stack_opt(key, dtype):
            if key in frames[0]:
                arr = torch.stack([torch.as_tensor(f[key]) for f in frames], dim=0).to(dtype=dtype)
                return arr.view(arr.size(0), -1)  # [T,1]
            return None

        # 稀疏 k 与 k_mask
        k_vals, m_vals = [], []
        for f in frames:
            if 'k' in f:
                kv = torch.as_tensor(f['k']).view(-1).float()
                if kv.numel() == 0 or torch.isnan(kv).any():
                    k_vals.append(torch.zeros(1, dtype=torch.float32))
                    m_vals.append(torch.zeros(1, dtype=torch.float32))
                else:
                    k_vals.append(kv[:1].float())
                    m_vals.append(torch.ones(1, dtype=torch.float32))
            else:
                k_vals.append(torch.zeros(1, dtype=torch.float32))
                m_vals.append(torch.zeros(1, dtype=torch.float32))
        pack['k'] = torch.stack(k_vals, dim=0)        # [T,1]
        pack['k_mask'] = torch.stack(m_vals, dim=0)   # [T,1]

        # 其他可选键
        for k, dt in (('T', torch.float32), ('V', torch.float32), ('P', torch.float32)):
            v = _stack_opt(k, dt)
            if v is not None:
                pack[k] = v
        st = _stack_opt('stage', torch.long)
        if st is not None:
            pack['stage'] = st.squeeze(-1).long()
        pack['filename'] = [f.get('filename', '') for f in frames]
        return pack

def collate_seq(batches):
    imgs = torch.stack([b['img'] for b in batches], dim=0)  # [B,T,C,H,W]
    out = {'img': imgs, 'filename': [b['filename'] for b in batches]}

    def _stack_opt(key, dtype):
        if key in batches[0]:
            v = torch.stack([b[key] for b in batches], dim=0).to(dtype)
            return v
        return None

    for k, dt in (('k', torch.float32), ('k_mask', torch.float32),
                  ('T', torch.float32), ('V', torch.float32), ('P', torch.float32)):
        v = _stack_opt(k, dt)
        if v is not None:
            out[k] = v
    st = _stack_opt('stage', torch.long)
    if st is not None:
        out['stage'] = st
    return out
