import re, sys, json, os
import numpy as np
import pandas as pd

TRAIN_XLSX = 'datasetone3/train/trainlabels.xlsx'
CFG_PATH   = 'models/swin_transformer/tiny_224_temporal.py'

def compute_stats(train_path):
    if not os.path.isfile(train_path):
        raise FileNotFoundError(train_path)
    df = pd.read_excel(train_path) if train_path.lower().endswith(('.xlsx','.xls')) else pd.read_csv(train_path)
    stats = {}
    for k in ['T','V','P']:
        if k not in df.columns:
            raise KeyError(f'Column "{k}" not found in {train_path}')
        s = pd.to_numeric(df[k], errors='coerce')
        stats[k] = (float(np.nanmean(s)), float(np.nanstd(s, ddof=0)))
    means = [stats['T'][0], stats['V'][0], stats['P'][0]]
    stds  = [stats['T'][1], stats['V'][1], stats['P'][1]]
    return means, stds

def replace_list_literal(text, var_name, values):
    # 替换形如 tvp_means = [ ... ] 的整行
    pat = rf'^(\s*{re.escape(var_name)}\s*=\s*)\[[^\]]*\](\s*)$'
    repl = rf'\1{json.dumps(values)}\2'
    return re.sub(pat, repl, text, flags=re.M)

def main():
    train = sys.argv[1] if len(sys.argv) > 1 else TRAIN_XLSX
    cfg   = sys.argv[2] if len(sys.argv) > 2 else CFG_PATH
    means, stds = compute_stats(train)
    print('Computed TVP stats from training set:')
    print('  tvp_means =', means)
    print('  tvp_stds  =', stds)

    if not os.path.isfile(cfg):
        raise FileNotFoundError(cfg)
    with open(cfg, 'r', encoding='utf-8') as f:
        text = f.read()
    bak = cfg + '.bak'
    with open(bak, 'w', encoding='utf-8') as f:
        f.write(text)

    text = replace_list_literal(text, 'tvp_means', means)
    text = replace_list_literal(text, 'tvp_stds', stds)

    with open(cfg, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Updated {cfg} and wrote backup to {bak}')

if __name__ == '__main__':
    main()
