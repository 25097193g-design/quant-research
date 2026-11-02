# P4c: MLP + stock embedding (torch, 本地MPS/CPU) —— 与LGBM blend
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 11_nn.py
# 结构: 数值特征(标准化) ⊕ stock嵌入(8维) → 128→64→1, log空间MSE, val RMSPE早停, 2种子平均
# 防泄漏: 标准化/中位数填充只用train折统计
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import rmspe

df = pd.read_parquet('cache/features_v3.parquet')
feat_cols = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')]
u = np.sort(df['time_id'].unique())
thr = u[int(len(u) * 0.8)]
tr, va = df[df.time_id < thr], df[df.time_id >= thr]

med = tr[feat_cols].median()
Xtr = tr[feat_cols].fillna(med).values.astype('float32')
Xva = va[feat_cols].fillna(med).values.astype('float32')
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
Str = tr['stock_id'].values.astype('int64')
Sva = va['stock_id'].values.astype('int64')
Ytr = np.log(tr['target'].values).astype('float32')
tgt = va['target'].values

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device={device} | 特征{len(feat_cols)}个 | train {len(tr):,} val {len(va):,}')

class Net(nn.Module):
    def __init__(self, n_num, n_stock, emb=8):
        super().__init__()
        self.emb = nn.Embedding(n_stock, emb)
        self.net = nn.Sequential(
            nn.Linear(n_num + emb, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))

    def forward(self, x, s):
        return self.net(torch.cat([x, self.emb(s)], 1)).squeeze(-1)

def train_seed(seed, epochs=60, bs=8192, patience=6):
    torch.manual_seed(seed)
    m = Net(len(feat_cols), int(Str.max()) + 1).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=8e-4)
    Xt = torch.from_numpy(Xtr).to(device)
    St = torch.from_numpy(Str).to(device)
    Yt = torch.from_numpy(Ytr).to(device)
    Xv, Sv = torch.from_numpy(Xva).to(device), torch.from_numpy(Sva).to(device)
    best, bad = (9.0, None), 0
    n = len(Xt)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = nn.functional.mse_loss(m(Xt[idx], St[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(Xv, Sv).cpu().numpy()
        s = rmspe(np.exp(np.clip(p, -20, 5)), tgt)
        if s < best[0]:
            best, bad = (s, np.clip(p, -20, 5).copy()), 0
        else:
            bad += 1
        if bad >= patience:
            break
    print(f'  seed={seed}: best RMSPE={best[0]:.4f} @epoch{ep}')
    return best[1]

preds = [train_seed(s) for s in (42, 2024)]  # 返回logit空间预测
p_nn = np.exp(np.mean(preds, axis=0))        # logit空间平均, 只exp一次(防下溢)
print(f'NN 2种子平均: RMSPE = {rmspe(p_nn, tgt):.4f}')

# ---- 与LGBM系blend(按键合并, 不赌行序) ----
nn_out = va[['stock_id', 'time_id']].copy()
nn_out['p_nn'] = p_nn
lg = pd.read_parquet('cache/twostage_val_pred.parquet')[['stock_id', 'time_id', 'p_b1', 'p_ts']]
mg = nn_out.merge(lg, on=['stock_id', 'time_id']).merge(
    va[['stock_id', 'time_id', 'target']], on=['stock_id', 'time_id'])
bl2 = np.exp(0.5 * np.log(mg.p_nn.values) + 0.5 * np.log(mg.p_b1.values))
bl3 = np.exp((np.log(mg.p_nn.values) + np.log(mg.p_b1.values) + np.log(mg.p_ts.values)) / 3)
print(f'blend(NN+B1):       RMSPE = {rmspe(bl2, mg.target.values):.4f}')
print(f'blend(NN+B1+两阶段): RMSPE = {rmspe(bl3, mg.target.values):.4f}')
mg.to_parquet('cache/nn_val_pred.parquet')
